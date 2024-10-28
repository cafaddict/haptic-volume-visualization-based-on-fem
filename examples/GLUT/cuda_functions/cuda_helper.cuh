#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>


#define cudaTextureType1D              0x01
#define cudaTextureType2D              0x02
#define cudaTextureType3D              0x03
#define cudaTextureTypeCubemap         0x0C
#define cudaTextureType1DLayered       0xF1
#define cudaTextureType2DLayered       0xF2
#define cudaTextureTypeCubemapLayered  0xFC






void wrapper(void);

void texture_test(void);



