#pragma once

#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &v) {
    return v * s;
}

namespace util {
    __host__ __device__ inline float length(const float3 &v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    __host__ __device__ inline float3 normalize(float3 v) {
        float len = length(v);
        return make_float3(v.x / len, v.y / len, v.z / len);
    }

    __host__ __device__ inline float dot(const float3 &a, const float3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
        return make_float3(
                a.y * b.z - b.y * a.z,
                a.z * b.x - b.z * a.x,
                a.x * b.y - b.x * a.y
            );
    }
}
