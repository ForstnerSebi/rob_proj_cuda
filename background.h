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


namespace backgroundfinder
{


    void start_rgb_background(string base_file, string compare_file, int delta)
    {
        start_time();
        int w_base, h_base, d_base, w_compare, h_compare, d_compare;
        cudaError_t cudaStatus;

        //defining size for threadblocks
        int blockSize = 256;

        //loading image and calculating dimaensions
        unsigned char* base_data_temp = stbi_load(base_file.c_str(), &w_base, &h_base, &d_base, 0);
        unsigned char* compare_data_temp = stbi_load(compare_file.c_str(), &w_compare, &h_compare, &d_compare, 0);
        int dim_base = w_base * h_base * d_base;
        int dim_comp = w_compare * h_compare * d_compare;

        log(1, "open_images", "finish open images for rgb background subtraction");

        //Allocate Unified Memory - for CPU and GPU
        //arrays for picture data
        uint8_t* base = 0, * comp = 0, * ret = 0;
        cudaStatus = cudaMallocManaged(&base, (dim_base * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of base_data failed!");
        }
        cudaStatus = cudaMallocManaged(&comp, (dim_comp * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of compare_data failed!");
        }
        cudaStatus = cudaMallocManaged(&ret, (w_base * h_base * 3 * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of pixels failed!");
        }

        log(1, "alloc_unified_mem", "finish allocate unified memory");

        if (base_data_temp != nullptr && w_base > 0 && h_base > 0 && compare_data_temp != nullptr &&
            w_compare > 0 && h_compare > 0 &&
            w_base == w_compare && h_base == h_compare && d_base == d_compare)
        {
            //copy from host to device memory
            cudaStatus = cudaMemcpy(base, base_data_temp, dim_base * sizeof(uint8_t), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            cudaStatus = cudaMemcpy(comp, compare_data_temp, dim_comp * sizeof(uint8_t), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            log(1, "copy_image_unified", "finish copy images to unified memory");

            //calculating the number of blocks required for full parallel processing ofimage
            int numBlocks = ((w_base * h_base) + blockSize - 1) / blockSize;



            //starting gpu kernel on numBlocks*blockSize threads
            background_rgb << <numBlocks, blockSize >> > (base, comp, w_base, h_base, d_base, ret, delta);

            log(0, "background_subtraction_rgb", "finish first background subtraction in rgb");

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
            stbi_write_png("objectfound_rgb.png", w_base, h_base, 3, ret, w_base * 3);
            log(1, "save_image", "finish save image");
        }
        else
        {
            std::cout << "The images are not comparable\n";
        }
        //Freeing unified memory used for previous steps
        //image data
        cudaFree(base);
        cudaFree(comp);
        cudaFree(ret);
        //Freeing memory used for loaded images; if done before this causes errors
        stbi_image_free(base_data_temp);
        stbi_image_free(compare_data_temp);
        log(1, "free_mem", "finish free all memory");
    }

    void start_hsv_background(string base_file, string compare_file, int delta)
    {
        start_time();
        int w_base, h_base, d_base, w_compare, h_compare, d_compare;
        cudaError_t cudaStatus;

        //defining size for threadblocks
        int blockSize = 256;

        //loading image and calculating dimaensions
        unsigned char* base_data_temp = stbi_load(base_file.c_str(), &w_base, &h_base, &d_base, 0);
        unsigned char* compare_data_temp = stbi_load(compare_file.c_str(), &w_compare, &h_compare, &d_compare, 0);
        int dim_base = w_base * h_base * d_base;
        int dim_comp = w_compare * h_compare * d_compare;

        log(1, "open_images", "finish open images for hsv background subtraction");

        //Allocate Unified Memory - for CPU and GPU
        //arrays for picture data
        float* base = 0, * comp = 0;
        uint8_t* ret = 0;
        cudaStatus = cudaMallocManaged(&base, (dim_base * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of base_data failed!");
        }
        cudaStatus = cudaMallocManaged(&comp, (dim_comp * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of compare_data failed!");
        }
        cudaStatus = cudaMallocManaged(&ret, (w_base * h_base * 3 * sizeof(uint8_t)));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc of pixels failed!");
        }

        log(1, "alloc_unified_mem", "finish allocate unified memory");

        //convert the image in hsv
        for (int i = 0;i < w_base * h_base; i++)
        {
            float h, s, v, r = base_data_temp[i * d_base], g = base_data_temp[i * d_base + 1], b = base_data_temp[i * d_base + 2];
            converter::RGBtoHSV(r, g, b, h, s, v);
            base[i * d_base] = h;
            base[i * d_base + 1] = s;
            base[i * d_base + 2] = v;
        }
        for (int i = 0;i < w_compare * h_compare; i++)
        {
            float h, s, v, r = compare_data_temp[i * d_base], g = compare_data_temp[i * d_base + 1], b = compare_data_temp[i * d_base + 2];
            converter::RGBtoHSV(r, g, b, h, s, v);
            comp[i * d_base] = h;
            comp[i * d_base + 1] = s;
            comp[i * d_base + 2] = v;
        }

        log(1, "convert-and-copy_image_unified", "finish converting image to hsv and copy in unified memory");

        if (base != nullptr && w_base > 0 && h_base > 0 && comp != nullptr &&
            w_compare > 0 && h_compare > 0 &&
            w_base == w_compare && h_base == h_compare && d_base == d_compare)
        {
            //calculating the number of blocks required for full parallel processing ofimage
            int numBlocks = ((w_base * h_base) + blockSize - 1) / blockSize;



            //starting gpu kernel on numBlocks*blockSize threads
            background_hsv << <numBlocks, blockSize >> > (base, comp, w_base, h_base, d_base, ret, delta);

            log(0, "background_subtraction_hsv", "finish first background subtraction in hsv");

            //checking for error while starting kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }

            //waiting for all threads to finish previous task; checking for errors
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            }

            log(0, "synchronize", "finish synchronize gpu threads");
            stbi_write_png("objectfound_hsv.png", w_base, h_base, 3, ret, w_base * 3);
            log(1, "save_image", "finish save image");

        }
        else
        {
            std::cout << "The images are not comparable\n";
        }

        //Freeing unified memory used for previous steps
        cudaFree(base);
        cudaFree(comp);
        cudaFree(ret);

        //Freeing memory used for loaded images; if done before this causes errors
        stbi_image_free(base_data_temp);
        stbi_image_free(compare_data_temp);
        log(1, "free_mem", "finish free all memory");
    }
}