#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simpleKernel(float *dst, float *src1, float *src2)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //float temp = src[idx];
  dst[idx] = src1[idx] + src2[idx];
}

int execute_uva(bool copy=false, bool print=false)
{
  float *src1,*src2, *dst;
  float *dsrc1, *dsrc2, *ddst, *dsrc2_1;
  size_t rsize = 256;
  size_t size = sizeof(float) * rsize * rsize;
  //cpu buffers
  src1 = (float *)malloc(size);
  src2 = (float *)malloc(size);
  dst = (float *)malloc(size);
  for (int i = 0; i < rsize * rsize; ++i) {
    src1[i] = (float)i;
    src2[i] = (float)(2 * i);
  }
  //gpu buffers
  cudaSetDevice(0);
  cudaDeviceEnablePeerAccess(1, 0);
  cudaMalloc(&ddst, size);
  cudaMalloc(&dsrc1, size);
  cudaMemcpy(dsrc1, src1, size, cudaMemcpyHostToDevice);
  // device setting here
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);
  cudaMalloc(&dsrc2, size);
  cudaMemcpy(dsrc2, src2, size, cudaMemcpyHostToDevice);
  
  //Launch the kernel
  cudaSetDevice(0);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  if (copy) { 
    //Add all the overhead of copying to the times
    //including memory allocation
    cudaMalloc(&dsrc2_1, size);
    cudaMemcpy(dsrc2_1, dsrc2, size, cudaMemcpyDefault);
    simpleKernel<<<rsize, rsize>>>(ddst, dsrc2_1, dsrc2);
  } else {
    simpleKernel<<<rsize, rsize>>>(ddst, dsrc1, dsrc2);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  cudaMemcpy(dst, ddst, size, cudaMemcpyDeviceToHost);
  if (print)
    for(int i = 0; i < rsize*rsize; ++i) {
      printf("%d: %f\n", i, dst[i]);
    }
  printf("Last item: %f\n", dst[rsize*rsize-1]);
  printf("Elapsed time: %f\n", time_ms); 
  // clean gpu buffers
  cudaFree(ddst);
  cudaFree(dsrc1);
  if (copy) cudaFree(dsrc2_1);
  // Just in case
  cudaSetDevice(1);
  cudaFree(dsrc2);
  // clean cpu buffers
  free(src1);
  free(src2);
  free(dst);
  return 0;
}


int main()
{
  int canAccess10, canAccess01;
  cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
  printf("Access status: %d\n", canAccess10);
  cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
  printf("Access status: %d\n", canAccess01);
  if (canAccess10 && canAccess01) {
    execute_uva(true);
  }
  return 0;
}
