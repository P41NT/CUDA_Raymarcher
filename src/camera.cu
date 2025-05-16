#include <cstdio>
#include <cuda_runtime.h>
#include "../include/camera.cuh"
#include "../include/utils.cuh"

#include <stdio.h>

namespace cam {
    Camera::Camera() {
        this->position = make_float3(0.0, 0.0, 10.0f);
        this->top = make_float3(0.0, 1.0f, 0.0f);
        this->right = make_float3(1.0f, 0.0f, 0.0f);
        this->front = make_float3(0.0f, 0.0f, -1.0f);

        this->yaw = 180.0f;
        this->pitch = 0.0f;
    }

    __host__ 
    void Camera::calculateRotation() {
        static const float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
        float3 newFront;
        newFront.z = cos(yaw * 3.14 / 180.0f) * cos(pitch * 3.14 / 180.0f);
        newFront.x = sin(yaw * 3.14 / 180.0f);
        newFront.y = sin(pitch * 3.14 / 180.0f);

        front = util::normalize(newFront);
        right = util::cross(front, worldUp);
        top = util::cross(front, right);
    }

    __host__ 
    void Camera::setPosition(float3 position) {
        this->position = position;
    }
    __host__ 
    void Camera::setPosition(float x, float y, float z) {
        this->position = util::normalize(make_float3(x, y, z));
    }

    __host__ 
    void Camera::moveForward(float speed) {
        position = position + front * speed;
    }
    __host__ 
    void Camera::moveBack(float speed) {
        position = position - front * speed;
    }
    __host__ 
    void Camera::moveRight(float speed) {
        position = position + right * speed;
    }
    __host__ 
    void Camera::moveLeft(float speed) {
        position = position - right * speed;
    }
    __host__ 
    void Camera::moveUp(float speed) {
        position = position + top * speed;
    }
    __host__ 
    void Camera::moveDown(float speed) {
        position = position - top * speed;
    }

    __host__
    void Camera::rotateRight(float sensitivity) {
        this->yaw -= sensitivity;
    }
    __host__
    void Camera::rotateLeft(float sensitivity) {
        yaw += sensitivity;
    }
    __host__
    void Camera::rotateUp(float sensitivity) {
        pitch += sensitivity;
    }
    __host__
    void Camera::rotateDown(float sensitivity) {
        pitch -= sensitivity;
    }

    __device__  __host__
    float3 Camera::getPosition() {
        return make_float3(position.x, position.y, position.z);
    }

    __device__ 
    float3 Camera::getRay(float2 screenCoord, float FOV) {
        float z = 1.0f / (tan((FOV / 2.0f) / 180 * 3.14f));
        float3 ray = screenCoord.x * right + screenCoord.y * top + front * z;
        return util::normalize(ray);
    }
}
