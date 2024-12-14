#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cmath>
#include <cstdlib>


using namespace std;
using namespace converter;
using namespace gpu_api;
using namespace logger;

namespace erode_dilate
{
    void start_erode(int radius, string base_file)
    {
        start_time();
        int w_base, h_base, d_base;
        cudaError_t cudaStatus;
        //defining size for threadblocks
        int blockSize = 256;

        //loading image and calculating dimaensions
        unsigned char* base_data_temp = stbi_load(base_file.c_str(), &w_base, &h_base, &d_base, 0);
        int dim_base = w_base * h_base * d_base;

        log(1, "open_image", "finish open images for erode");

        //Allocate Unified Memory - for CPU and GPU
        //arrays for picture data
        uint8_t* base = 0, * ret = 0;
        cudaStatus = cudaMallocManaged(&base, (dim_base * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of base_data failed!");
        }
        cudaStatus = cudaMallocManaged(&ret, (w_base * h_base * 3 * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of pixels failed!");
        }

        log(1, "alloc_unified_mem", "finish allocate unified memory");

        if (base_data_temp != nullptr && w_base > 0 && h_base > 0 && d_base > 0)
        {
            //copy from host to unified memory
            cudaStatus = cudaMemcpy(base, base_data_temp, dim_base * sizeof(uint8_t), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            log(1, "copy_image_unified", "finish copy image to unified memory");

            //calculating the number of blocks required for full parallel processing ofimage
            int numBlocks = ((w_base * h_base) + blockSize - 1) / blockSize;



            //starting gpu kernel on numBlocks*blockSize threads
            erode << <numBlocks, blockSize >> > (base, w_base, h_base, d_base, radius, ret);

            log(0, "erode", "finish first erode");

            //checking for error while starting kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %d\n", cudaStatus);
            }

            //waiting for all threads to finish previous task; checking for errors
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            }

            log(0, "synchronize", "finish synchronize gpu threads");
            stbi_write_png("erode.png", w_base, h_base, 3, ret, w_base * 3);
            log(1, "save_image", "finish save image");
        }
        else
        {
            std::cout << "The images are not comparable\n";
        }
        //Freeing unified memory used for previous steps
        //image data
        cudaFree(base);
        cudaFree(ret);
        //Freeing memory used for loaded images; if done before this causes errors
        stbi_image_free(base_data_temp);
        log(1, "free_mem", "finish free all memory");
    }

    void start_dilate(int radius, string base_file)
    {
        start_time();
        int w_base, h_base, d_base;
        cudaError_t cudaStatus;
        //defining size for threadblocks
        int blockSize = 256;

        //loading image and calculating dimaensions
        unsigned char* base_data_temp = stbi_load(base_file.c_str(), &w_base, &h_base, &d_base, 0);
        int dim_base = w_base * h_base * d_base;

        log(1, "open_image", "finish open images for dilate");

        //Allocate Unified Memory - for CPU and GPU
        //arrays for picture data
        uint8_t* base = 0, * ret = 0;
        cudaStatus = cudaMallocManaged(&base, (dim_base * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of base_data failed!");
        }
        cudaStatus = cudaMallocManaged(&ret, (w_base * h_base * 3 * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of pixels failed!");
        }

        log(1, "alloc_unified_mem", "finish allocate unified memory");

        if (base_data_temp != nullptr && w_base > 0 && h_base > 0 && d_base > 0)
        {
            //copy from host to unified memory
            cudaStatus = cudaMemcpy(base, base_data_temp, dim_base * sizeof(uint8_t), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            log(1, "copy_image_unified", "finish copy image to unified memory");

            //calculating the number of blocks required for full parallel processing ofimage
            int numBlocks = ((w_base * h_base) + blockSize - 1) / blockSize;


            //starting gpu kernel on numBlocks*blockSize threads
            dilate << <numBlocks, blockSize >> > (base, w_base, h_base, d_base, radius, ret);

            log(0, "dilate", "finish first dilate");

            //checking for error while starting kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %d\n", cudaStatus);
            }

            //waiting for all threads to finish previous task; checking for errors
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            }

            log(0, "synchronize", "finish synchronize gpu threads");
            stbi_write_png("dilate.png", w_base, h_base, 3, ret, w_base * 3);
            log(1, "save_image", "finish save image");
        }
        else
        {
            std::cout << "The images are not comparable\n";
        }
        //Freeing unified memory used for previous steps
        //image data
        cudaFree(base);
        cudaFree(ret);
        //Freeing memory used for loaded images; if done before this causes errors
        stbi_image_free(base_data_temp);
        log(1, "free_mem", "finish free all memory");
    }
}
