#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <raylib.h>
#include <time.h>

#include "../include/raymarcher.cuh"

#define RAYGUI_IMPLEMENTATION
#include "../external/raygui.h"

#include <iostream>

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

    float sensitivity = 0.1f;
    float turnSensitivity = 0.1f;

    float power_host = 8.0f;
    float darkness_host = 60.0f;
    float cutoff_host = 16.0f;

    float3 colorMixA_host = {1.0f, 0.0f, 0.0f};
    float3 colorMixB_host = {0.0f, 0.0f, 1.0f};

    camera->calculateRotation();
    camera->setPosition(-2.32173, -1.8306, 0.893631);

    while (!WindowShouldClose()) {
        const dim3 blockSize(16, 16);
        const dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

        cudaDeviceSynchronize();

        if (IsKeyDown(KEY_W)) camera->moveForward(sensitivity);
        if (IsKeyDown(KEY_S)) camera->moveBack(sensitivity);
        if (IsKeyDown(KEY_A)) camera->moveLeft(sensitivity);
        if (IsKeyDown(KEY_D)) camera->moveRight(sensitivity);

        if (IsKeyDown(KEY_E)) camera->moveDown(sensitivity);
        if (IsKeyDown(KEY_Q)) camera->moveUp(sensitivity);
        
        if (IsKeyDown(KEY_L)) camera->rotateRight(turnSensitivity);
        if (IsKeyDown(KEY_H)) camera->rotateLeft(turnSensitivity);
        if (IsKeyDown(KEY_K)) camera->rotateUp(turnSensitivity);
        if (IsKeyDown(KEY_J)) camera->rotateDown(turnSensitivity);

        camera->calculateRotation();

        std::cerr << "Camera position: " << camera->getPosition().x << " " << camera->getPosition().y << " " << camera->getPosition().z << std::endl;

        updateStuff<<<1, 1>>>(power_host, darkness_host, cutoff_host, colorMixA_host, colorMixB_host);
        cudaDeviceSynchronize();
        render<<<gridSize, blockSize>>>(canvas, *camera, clock());

        cudaDeviceSynchronize();
        cudaMemcpy(hostCanvas, canvas, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost);

        UpdateTexture(canvasTexture, hostCanvas);

        if (IsKeyPressed(KEY_P)) {
            Image image = LoadImageFromTexture(canvasTexture);
            ExportImage(image, "saved_image.png");
        }

        BeginDrawing();
        ClearBackground(WHITE);
        DrawTexture(canvasTexture, 0, 0, WHITE);

        GuiSliderBar((Rectangle){100, 100, 200, 20}, "Low", "High", &sensitivity, 0.0f, 0.5f);
        GuiSliderBar((Rectangle){100, 130, 200, 20}, "Low", "High", &turnSensitivity, 0.0f, 2.0f);

        GuiSliderBar((Rectangle){100, 160, 200, 20}, "Low", "High", &power_host, 3.0f, 10.0f);
        GuiSliderBar((Rectangle){100, 190, 200, 20}, "Low", "High", &darkness_host, 1.0f, 80.0f);
        GuiSliderBar((Rectangle){100, 210, 200, 20}, "Low", "High", &cutoff_host, 0.0f, 200.0f);

        GuiSliderBar((Rectangle){400, 100, 200, 20}, "Low", "High", &colorMixA_host.x, 0.0f, 1.0f);
        GuiSliderBar((Rectangle){400, 130, 200, 20}, "Low", "High", &colorMixA_host.y, 0.0f, 1.0f);
        GuiSliderBar((Rectangle){400, 160, 200, 20}, "Low", "High", &colorMixA_host.z, 0.0f, 1.0f);

        GuiSliderBar((Rectangle){400, 190, 200, 20}, "Low", "High", &colorMixB_host.x, 0.0f, 1.0f);
        GuiSliderBar((Rectangle){400, 220, 200, 20}, "Low", "High", &colorMixB_host.y, 0.0f, 1.0f);
        GuiSliderBar((Rectangle){400, 250, 200, 20}, "Low", "High", &colorMixB_host.z, 0.0f, 1.0f);

        EndDrawing();
    }

    cudaFree(canvas);

    UnloadTexture(canvasTexture);
    CloseWindow();

    return 0;
}
