#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace gpu_api
{
    __global__ void background_rgb(uint8_t* base, uint8_t* compare, int w, int h, int d, uint8_t* pixels, int dif)
    {

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;


        if (index < (w * h))
        {
            for (int p = index; p < (w * h); p += stride)
            {

                float delta = (abs(base[p * d + 0] - compare[p * d + 0]) / 3 + \
                    abs(base[p * d + 1] - compare[p * d + 1]) / 3 + \
                    abs(base[p * d + 2] - compare[p * d + 2]) / 3);
                if (delta > dif)
                {
                    pixels[p * 3 + 0] = 255;
                    pixels[p * 3 + 1] = 255;
                    pixels[p * 3 + 2] = 255;
                }
                else
                {
                    pixels[p * 3 + 0] = 0;
                    pixels[p * 3 + 1] = 0;
                    pixels[p * 3 + 2] = 0;
                }
            }
        }
    }

    __global__ void background_hsv(float* base, float* compare, int w, int h, int d, uint8_t* pixels, int dif)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        if (index < (w * h))
        {
            for (int p = index; p < (w * h); p += stride)
            {
                int delta = (abs(base[p * d + 0] - compare[p * d + 0]) / 2 + \
                    abs(base[p * d + 1] - compare[p * d + 1]) / 2);
                if (delta > dif)
                {
                    pixels[p * 3 + 0] = 255;
                    pixels[p * 3 + 1] = 255;
                    pixels[p * 3 + 2] = 255;
                }
                else
                {
                    pixels[p * 3 + 0] = 0;
                    pixels[p * 3 + 1] = 0;
                    pixels[p * 3 + 2] = 0;
                }
            }
        }
    }

    //dilate: maximum
    __global__ void dilate(uint8_t* image, int w, int h, int d, int r, uint8_t* pixels)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        if (index < (w * h))
        {
            for (int p = index; p < (w * h); p += stride) //alle Pixel
            {
                int max_pixel = (image[(p)*d] + image[(p)*d + 1] + image[(p)*d + 2]);
                for (int i = -1 * r; i <= r; i++) //LINKS-RECHTS
                {
                    for (int j = -1 * r; j <= r; j++) //OBEN-UNTEN
                    {
                        int pixel = p + i + j * w;
                        if (pixel >= 0 && pixel < (w * h) && (p % w) >= (-1 * i) && (p % w) + i < w)
                        {
                            max_pixel = max_pixel + (image[(pixel)*d] + image[(pixel)*d + 1] + image[(pixel)*d + 2]);
                        }
                    }
                }

                if (max_pixel)
                {
                    pixels[p * 3 + 0] = 255;
                    pixels[p * 3 + 1] = 255;
                    pixels[p * 3 + 2] = 255;
                }
                else
                {
                    pixels[p * 3 + 0] = 0;
                    pixels[p * 3 + 1] = 0;
                    pixels[p * 3 + 2] = 0;
                }
            }
        }
    }

    //erode: minimum
    __global__ void erode(uint8_t* image, int w, int h, int d, int r, uint8_t* pixels)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        if (index < (w * h))
        {
            for (int p = index; p < (w * h); p += stride) //alle Pixel
            {
                int min_pixel = ((255 - image[(p)*d]) + (255 - image[(p)*d + 1]) + (255 - image[(p)*d + 2]));
                for (int i = -1 * r; i <= r; i++) //LINKS-RECHTS
                {
                    for (int j = -1 * r; j <= r; j++) //OBEN-UNTEN
                    {
                        int pixel = p + i + j * w;
                        if (pixel >= 0 && pixel < (w * h) && (p % w) >= (-1 * i) && (p % w) + i < w)
                        {
                            min_pixel = min_pixel + ((255 - image[(pixel)*d]) + (255 - image[(pixel)*d + 1]) + (255 - image[(pixel)*d + 2]));
                        }
                    }
                }

                if (!min_pixel)
                {
                    pixels[p * 3 + 0] = 255;
                    pixels[p * 3 + 1] = 255;
                    pixels[p * 3 + 2] = 255;
                }
                else
                {
                    pixels[p * 3 + 0] = 0;
                    pixels[p * 3 + 1] = 0;
                    pixels[p * 3 + 2] = 0;
                }
            }
        }
    }
}
