#pragma onoce

__device__ float sphereSDF(float3 p, float radius, float3 center);

__device__ float mandelblubSDF(float3 p, int *steps);

__device__ float scene(float3 p, int *steps = NULL);

__device__ float3 getNormal(float3 p);

__device__ float2 raymarch(float3 position, float3 direction, int *steps);

__global__ void render(uchar4* canvas, cam::Camera& cam, float currTime);

__host__ void renderScene(uchar4* canvas, cam::Camera& cam, float currTime) {
    dim3 blockSize(16, 16);
    dim3 gridSize((cam.width + blockSize.x - 1) / blockSize.x, (cam.height + blockSize.y - 1) / blockSize.y);
    render<<<gridSize, blockSize>>>(canvas, cam, currTime);
}
