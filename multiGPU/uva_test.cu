#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

/*__global__ void vectorAdd(float *dst, float *v1, float *v2)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = v1[idx] + v2[idx];
}


int execute()
{
  float *v0, *v1, *dst;
  float *dst_host, *v0_host, *v1_host;
  size_t size = sizeof(float)*64*64;
  //cudaSetDevice(0);
  dst_host = (float *)malloc(size);
  cudaMalloc(&v0, size);
  cudaMemset(v0, 1.0, size);
  
  //cudaSetDevice(1);
  cudaMalloc(&dst, size);
  cudaMalloc(&v1, size);
  cudaMemset(v1, 2.0, size);
  cudaDeviceSynchronize();
  //Enable the peer access from 1 to 0
  //cudaDeviceEnablePeerAccess(1, 0);
  //Launch kernel in 1
  vectorAdd<<<64, 64>>>(dst, v0, v1);
  //Take result to the cpu
  cudaMemcpy(dst_host, dst, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  float sum = 0.0;
  for (int i = 0; i < 64*64; ++i)
    sum += dst_host[i];
  if (sum == 64*64*3) {
    printf("Passed...\n");
  } else {
    printf("Sum: %f\n", sum);
    printf("Failed...\n");
  }
  // Clean up
  cudaFree(v0);
  cudaFree(v1);
  cudaFree(dst);
  free(dst_host);
  return 0;
  }*/

  __global__ void simpleKernel(float *dst, float *src1, float *src2)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //float temp = src[idx];
  dst[idx] = src1[idx] + src2[idx];
}

int execute()
{
  float *src1,*src2, *dst;
  float *dsrc1, *dsrc2, *ddst;
  size_t rsize = 8;
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
  simpleKernel<<<rsize, rsize>>>(ddst, dsrc1, dsrc2);
  cudaMemcpy(dst, ddst, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < rsize*rsize; ++i) {
    printf("%d: %f\n", i, dst[i]);
  }
  cudaFree(ddst);
  cudaFree(dsrc1);
  cudaFree(dsrc2);
  free(src1);
  free(src2);
  free(dst);
  return 0;
}

int main()
{
  int canAccess;
  cudaDeviceCanAccessPeer(&canAccess, 1, 0);
  printf("Access status: %d\n", canAccess);
  cudaDeviceCanAccessPeer(&canAccess, 0, 1);
  printf("Access status: %d\n", canAccess);
  if (canAccess) {
    execute();
  }
  return 0;
}
