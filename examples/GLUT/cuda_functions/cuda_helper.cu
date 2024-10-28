

//#include "cuda_helper.cuh"
#include "cuda_header.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>




//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>

texture<float, 3, cudaReadModeElementType> texPtr;

__global__ void test_kernel(void) {
	printf("Hello, world! from GPU\n");
}


__global__ void texture_kernel(float *dst, int width, int height)
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = blockIdx.x;

	float sample = tex3D(texPtr, x, y, z);

	dst[z * width * height + y * width + x] = sample;

}


void wrapper(void)
{
	test_kernel << <1, 1 >> > ();
	printf("Hello, world!\n");
	cudaDeviceSynchronize();
}

void texture_test(void)
{
	int width = 2;
	int height = 2;
	int depth = 2;
	float *src = (float*)malloc(sizeof(float)*width*height*depth);
	float *dst = (float*)malloc(sizeof(float)*width*height*depth);
	float* src_d;
	float* dst_d;

	cudaError_t result;
	cudaArray * cu_array;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaExtent extent;
	extent.width = width;
	extent.height = height;
	extent.depth = depth;

	result = cudaMalloc((void**)&dst_d, sizeof(float)*width*height*depth);
	result = cudaMalloc3DArray(&cu_array, &channelDesc, extent, 0);
	if (result != cudaSuccess) {
		fprintf(stderr, "Texture3D - failed to malloc 3D array - %s \n", cudaGetErrorString(result));
		return;
	}

	for (int i = 0; i < width*height*depth; i++) {
		src[i] = i;
	}

	cudaMemcpy3DParms params;

	memset(&params, 0, sizeof(params));
	params.srcPtr.pitch = sizeof(float)* width;
	params.srcPtr.ptr = src;
	params.srcPtr.xsize = width;
	params.srcPtr.ysize = height;

	params.srcPos.x = 0;
	params.srcPos.y = 0;
	params.srcPos.z = 0;

	params.dstArray = cu_array;

	params.dstPos.x = 0;
	params.dstPos.y = 0;
	params.dstPos.z = 0;

	params.extent.width = width;
	params.extent.depth = depth;
	params.extent.height = height;

	params.kind = cudaMemcpyHostToDevice;

	result = cudaMemcpy3D(&params);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D - failed to copy from host buffer to device array - %s\n", cudaGetErrorString(result));
		return;
	}

	texPtr.addressMode[0] = cudaAddressModeWrap;
	texPtr.addressMode[1] = cudaAddressModeWrap;
	texPtr.addressMode[2] = cudaAddressModeWrap;
	texPtr.filterMode = cudaFilterModePoint;// cudaFilterModePoint or cudaFilterModeLinear
	texPtr.normalized = false;

	// bind to array
	result = cudaBindTextureToArray(texPtr, cu_array, channelDesc);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaBindTextureToArray() - failed to bind texture to array - %s", cudaGetErrorString(result));
		return;
	}

	texture_kernel << <dim3(depth, 1), dim3(width, height) >> > (dst_d, width, height);

	cudaThreadSynchronize();
	result = cudaMemcpy(dst, dst_d, sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost);

	for (int i = 0; i < width*height*depth; i++) {
		printf("%d %.1f == %.1f \n", i, src[i], dst[i]);
	}printf("\n\n");

	// Texture 메모리 내용 갱신하기
	float *srcSingleChannel = (float*)malloc(sizeof(float)*width*height);
	for (int i = 0; i < width*height; i++) {
		srcSingleChannel[i] = 100 + i;
	}

	params.srcPtr.ptr = srcSingleChannel;
	params.dstPos.z = 1;
	params.extent.depth = 1;
	result = cudaMemcpy3D(&params);

	texture_kernel << <dim3(depth, 1), dim3(width, height) >> > (dst_d, width, height);

	cudaThreadSynchronize();
	result = cudaMemcpy(dst, dst_d, sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D - failed to copy from host buffer to device array - %s\n", cudaGetErrorString(result));
		return;
	}

	for (int i = 0; i < width*height*depth; i++) {
		printf("%d %.1f == %.1f \n", i, src[i], dst[i]);
	}
	cudaDeviceSynchronize();


}



//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
