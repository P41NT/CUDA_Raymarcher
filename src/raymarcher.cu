#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <raylib.h>

#include "../include/camera.cuh"
#include "../include/raymarcher.cuh"

#include <stdio.h>

#include <time.h>

__device__ float power = 8;
__device__ float darkness = 60.0f;
__device__ float cutoff = 16.0f;
__device__ float blackAndWhite = 0.0f;

__device__ float3 colorMixA = {0.1f, 0.0f, 1.0f};
__device__ float3 colorMixB = {0.0f, 0.0f, 0.1f};

__device__ float sphereSDF(float3 p, float radius, float3 center) {
    float3 diff = p - center;
    return util::length(diff) - radius;
}

__device__ float mandelblubSDF(float3 p, int *steps) {
    float3 z = p;
    float dr = 1.0f;
    float r = 0.0f;

    for (int i = 0; i < 20; i++) {
        r = util::length(z);
        if (steps != NULL) *steps = i;
        if (r > 2.0f) break;

        float theta = acos(z.z / r) * power;
        float phi = atan2(z.y, z.x) * power;
        dr = pow(r, power - 1) * power * dr  + 1.0f;

        float zr = pow(r, power);

        z = zr * make_float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z = z + p;
    }

    return 0.5 * log(r) * r / dr;
}

__device__ float scene(float3 p, int *steps = NULL) {
    return mandelblubSDF(p, steps);
}

__device__ float3 getNormal(float3 p) {
    float3 normal = {0.0f, 0.0f, 0.0f};
    const float delta = 0.0001f;

    normal.x = scene(p + make_float3(delta, 0.0f, 0.0f)) - scene(p - make_float3(delta, 0.0f, 0.0f));
    normal.y = scene(p + make_float3(0.0f, delta, 0.0f)) - scene(p - make_float3(0.0f, delta, 0.0f));
    normal.z = scene(p + make_float3(0.0f, 0.0f, delta)) - scene(p - make_float3(0.0f, 0.0f, delta));

    return util::normalize(normal);
}

__device__ float2 raymarch(float3 position, float3 direction, int *steps) {
    float currDistance = 0;

    for (int i = 0; i < MAX_STEPS; i++) {
        float3 newPosition = position + (direction * currDistance);
        float dist = scene(newPosition, steps);

        if (currDistance > MAX_DIST) return {currDistance, (float)i};
        if (dist < EPSILON) return {currDistance, (float)i};

        currDistance += dist;
    }

    return {MAX_DIST, (float)MAX_STEPS};
}

__device__ inline float sharp_exponential(float x, float k = 0.5f) {
    return (expf(k * x) - 1.0f) / (expf(k) - 1.0f);
}

__global__ void render(uchar4* canvas, cam::Camera& cam, float currTime) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > WIDTH || y > HEIGHT) return;

    float2 uv = make_float2((float)x / WIDTH, (float)y / HEIGHT);
    uv.x -= 0.5, uv.y -= 0.5;

    uv.x *= ASPECT_RATIO;

    float3 cameraPosition = cam.getPosition();
    float3 rayDirection = cam.getRay(uv, 60.0f);

    int fractalSteps = 0;
    float2 temp = raymarch(cameraPosition, rayDirection, &fractalSteps);
    float dist = temp.x;
    float steps = temp.y;

    float3 finalRay = cameraPosition + dist * rayDirection;

    float3 lightDirection = make_float3(0.0f, -1.0f, 0.0f);

    float3 background = util::lerp(
        make_float3(51.0f, 3.0f, 20.0f),
        make_float3(16.0f, 6.0f, 28.0f),
        uv.y + 0.5f
    );

    if (dist < MAX_DIST) {
        float3 normal = getNormal(finalRay);

        float colorA = util::saturate(util::dot(make_float3(0.5, 0.5, 0.5) + normal*0.5f, (-1) * lightDirection));
        float colorB = util::saturate(steps / cutoff);

        float3 color = util::saturateV(colorA * colorMixA + colorB * colorMixB);
        float brightness = util::length(color);
        color = color + (1.0f - brightness) * make_float3(0.0f, 0.0f, 0.0f);

        canvas[y * WIDTH + x] = {
            (unsigned char)(255.0f * color.x),
            (unsigned char)(255.0f * color.y),
            (unsigned char)(255.0f * color.z),
            255
        };

    }
    else {
        float rim  = (steps / darkness);
        float3 color = util::lerp(background, make_float3(255.0f, 255.0f, 255.0f), blackAndWhite) * rim;
        float brightness = util::length(color);
        canvas[y * WIDTH + x] = {
            (unsigned char)(color.x),
            (unsigned char)(color.y),
            (unsigned char)(color.z),
            255
        };
    }
}

__global__ void updateStuff(float power_host, float darkness_host, float cutoff_host, 
                            float3 colorMixA_host, float3 colorMixB_host) {
    power = power_host;
    darkness = darkness_host;
    cutoff = cutoff_host;

    colorMixA = colorMixA_host;
    colorMixB = colorMixB_host;
}

__device__ void boxBlur(uchar4* canvas, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > WIDTH || y > HEIGHT) return;

    int count = 0;
    float3 sum = {0.0f, 0.0f, 0.0f};

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                uchar4 neighborColor = canvas[ny * WIDTH + nx];
                sum = sum + make_float3(neighborColor.x, neighborColor.y, neighborColor.z);
                count++;
            }
        }
    }

    if (count > 0) {
        sum = sum / (float)count;
        canvas[y * WIDTH + x] = {
            (unsigned char)(sum.x),
            (unsigned char)(sum.y),
            (unsigned char)(sum.z),
            255
        };
    }
}
