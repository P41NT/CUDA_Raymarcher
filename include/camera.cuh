#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>

namespace cam {
    class Camera {
        float3 position;

        float3 top, front, right;

        float yaw, pitch;
    public:
        Camera();

        __host__ void setPosition(float3 position);
        __host__ void setPosition(float x, float y, float z);

        __host__ void moveForward(float speed);
        __host__ void moveBack(float speed);
        __host__ void moveRight(float speed);
        __host__ void moveLeft(float speed);
        __host__ void moveUp(float speed);
        __host__ void moveDown(float speed);

        __host__ void rotateRight(float sensitivity);
        __host__ void rotateLeft(float sensitivity);
        __host__ void rotateUp(float sensitivity);
        __host__ void rotateDown(float sensitivity);

        __host__ void calculateRotation();

        __device__ float3 getRay(float2 screenCoord, float FOV);
        __host__ __device__ float3 getPosition();
    };
}
