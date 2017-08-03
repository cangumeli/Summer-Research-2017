#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simpleKernel(float *dst, float *src)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float temp = src[idx];
  dst[idx] = temp * temp;
}

int execute()
{
  float *src, *dst;
  float *dsrc, *ddst;
  size_t rsize = 64;
  size_t size = sizeof(float) * 64;
  //cpu buffers
  src = (float *)malloc(size);
  dst = (float *)malloc(size);
  //gpu buffers
  cudaMalloc(&dsrc, size);
  cudaMalloc(&ddst, size);
  for (int i = 0; i < 64; ++i) {
    src[i] = (float)i;
  }
  /*for (int i = 0; i < 64; ++i) {
    printf("%f\n", src[i]);
    }*/
  cudaMemcpy(dsrc, src, size, cudaMemcpyHostToDevice);
  simpleKernel<<<1, 64>>>(ddst, dsrc);
  cudaMemcpy(dst, ddst, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 64; ++i) {
    printf("%d: %f\n", i, dst[i]);
  }
  cudaFree(ddst);
  cudaFree(dsrc);
  free(src);
  free(dst);
  return 0;
}

int main()
{
  printf("\nSetting device...");
  cudaSetDevice(0);
  execute();
  printf("\nSetting device...");
  cudaSetDevice(1);
  execute();
  return 0;
}
