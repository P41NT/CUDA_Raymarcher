#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <raylib.h>

#include "../include/camera.cuh"
#include "../include/utils.cuh"

#include <time.h>

const float MAX_DIST = 100.0f;
const int MAX_STEPS = 120.0f;

const int WIDTH = 1600;
const int HEIGHT = 1600;

const float ASPECT_RATIO = (float) WIDTH / HEIGHT;

const float EPSILON = 0.0001f;

const float power = 8;


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

    if (dist < MAX_DIST) {

        float diffuse = sharp_exponential(fractalSteps / 20.0f, 0.3f);
        float3 color = diffuse * make_float3(0.9f, 0.1f, 0.7f);

        canvas[y * WIDTH + x] = {
            (unsigned char)(255.0f * color.x),
            (unsigned char)(255.0f * color.y),
            (unsigned char)(255.0f * color.z),
            255
        };

    }
    else {
        float brightness = steps / MAX_STEPS;
        float3 color = brightness * make_float3(0.9f, 0.1f, 0.8f);
        canvas[y * WIDTH + x] = {
            (unsigned char)(255.0f * color.x),
            (unsigned char)(255.0f * color.y),
            (unsigned char)(255.0f * color.z),
            255
        };
    }
}

int main() {
    uchar4 *canvas;
    cudaMallocManaged(&canvas, WIDTH * HEIGHT * sizeof(uchar4));

    InitWindow(WIDTH, HEIGHT, "skibidi cuda raymarcher");

    Image image = {
        .data = malloc(WIDTH * HEIGHT * sizeof(uchar4)),
        .width = WIDTH,
        .height = HEIGHT,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };

    Texture2D canvasTexture = LoadTextureFromImage(image);
    free(image.data); 

    cam::Camera *camera = new cam::Camera();

    uchar4 *hostCanvas = (uchar4 *)malloc(WIDTH * HEIGHT * sizeof(uchar4));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        hostCanvas[i] = (uchar4){ 0, 0, 0, 255 };  
    }

    camera->calculateRotation();

    while (!WindowShouldClose()) {
        const dim3 blockSize(16, 16);
        const dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

        cudaDeviceSynchronize();

        if (IsKeyDown(KEY_W)) camera->moveForward(0.005f);
        if (IsKeyDown(KEY_S)) camera->moveBack(0.005f);
        if (IsKeyDown(KEY_A)) camera->moveLeft(0.005f);
        if (IsKeyDown(KEY_D)) camera->moveRight(0.005f);

        if (IsKeyDown(KEY_E)) camera->moveDown(0.005f);
        if (IsKeyDown(KEY_Q)) camera->moveUp(0.005f);
        
        if (IsKeyDown(KEY_L)) camera->rotateRight(0.05f);
        if (IsKeyDown(KEY_H)) camera->rotateLeft(0.05f);
        if (IsKeyDown(KEY_K)) camera->rotateUp(0.05f);
        if (IsKeyDown(KEY_J)) camera->rotateDown(0.05f);

        camera->calculateRotation();

        render<<<gridSize, blockSize>>>(canvas, *camera, 0.000002 * clock());

        cudaDeviceSynchronize();
        cudaMemcpy(hostCanvas, canvas, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost);

        UpdateTexture(canvasTexture, hostCanvas);

        BeginDrawing();
        ClearBackground(WHITE);
        DrawTexture(canvasTexture, 0, 0, WHITE);
        EndDrawing();
    }

    cudaFree(canvas);

    UnloadTexture(canvasTexture);
    CloseWindow();

    return 0;
}
