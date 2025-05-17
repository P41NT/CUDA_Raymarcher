#pragma onoce

#include "camera.cuh"

__global__ void updateStuff(float power_host, float darkness_host, float cutoff_host, 
                            float3 colorMixA_host, float3 colorMixB_host);
__global__ void render(uchar4* canvas, cam::Camera& cam, float currTime);
__device__ void boxBlur(uchar4* canvas, int radius);
