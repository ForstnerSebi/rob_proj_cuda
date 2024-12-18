#include <stdio.h>

#include <iostream>
#include <cmath>
#include <cstdlib>

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
}
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "logger.h"

#include "gpu_api.h"

#include "RGB-HSV_conversion.h"
#include "background.h"
#include "erode_dilate.h"


using namespace std;
using namespace gpu_api;
using namespace backgroundfinder;
using namespace erode_dilate;




int main(int argc, char** argv)
{
    string base_image = "base.png", compare_image = "compare.png";
    int radius = 2, delta = 25;
    if(argc >1)
    {
        if ((string)argv[1] == "--rgb" || (string)argv[1] == "-r")
        {
            switch (argc)
            {
            case 2:
                break;
            case 3:
                base_image = (string)argv[2];
                break;
            case 4:
                base_image = (string)argv[2];
                compare_image = (string)argv[3];
                break;
            default:
                base_image = (string)argv[2];
                compare_image = (string)argv[3];
                delta = atoi(argv[4]);
                break;
            }
            start_rgb_background(base_image, compare_image, delta);
        }
        else if ((string)argv[1] == "--hsv" || (string)argv[1] == "-h")
        {
            switch (argc)
            {
            case 2:
                break;
            case 3:
                base_image = (string)argv[2];
                break;
            case 4:
                base_image = (string)argv[2];
                compare_image = (string)argv[3];
                break;
            default:
                base_image = (string)argv[2];
                compare_image = (string)argv[3];
                delta = atoi(argv[4]);
                break;
            }
            start_hsv_background(base_image, compare_image, delta);
        }
        else if ((string)argv[1] == "--erode" || (string)argv[1] == "-e")
        {
            switch (argc)
            {
            case 2:
                base_image = "objectfound_rgb.png";
                break;
            case 3:
                base_image = (string)argv[2];
                break;
            default:
                base_image = (string)argv[2];
                radius = atoi(argv[3]);
                break;
            }
            start_erode(radius, base_image);
        }
        else if ((string)argv[1] == "--dilate" || (string)argv[1] == "-d")
        {
            switch (argc)
            {
            case 2:
                base_image = "objectfound_rgb.png";
                break;
            case 3:
                base_image = (string)argv[2];
                break;
            default:
                base_image = (string)argv[2];
                radius = atoi(argv[3]);
                break;
            }
            start_dilate(radius, base_image);
        }
        else if ((string)argv[1] == "help" || (string)argv[1] == "--help" || (string)argv[1] == "-h")
        {
            cout << "possible commands: \n\n" << "rgb <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on rbg\n" << "hsv <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on hsv\n" <<
                "erode <<pic1>> <<radius>> \t\t\tuses erode on the picture with given radius\n" << "dilate <<pic1>> <<radius>> \t\t\tuses dilate on the picture with given radius\n";
        }
        else
        {
            cout << "please use one of the following commands: \n\n" << "rgb <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on rbg\n" << "hsv <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on hsv\n" <<
                "erode <<pic1>> <<radius>> \t\t\tuses erode on the picture with given radius\n" << "dilate <<pic1>> <<radius>> \t\t\tuses dilate on the picture with given radius\n" << "help \t\t\t\t\t\tshows this list of commands\n";
        }
    }
    else
    {
        cout << "please use one of the following commands: \n\n" << "rgb <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on rbg\n" << "hsv <<pic1>> <<pic2>> <<delta>> \t\tfinds objects based on hsv\n" <<
            "erode <<pic1>> <<radius>> \t\t\tuses erode on the picture with given radius\n" << "dilate <<pic1>> <<radius>> \t\t\tuses dilate on the picture with given radius\n" << "help \t\t\t\t\t\tshows this list of commands\n";
    }

}

