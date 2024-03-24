
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32*1024*1024    // 输入数据的长度
#define BLOCK_SIZE 256    // 每个block的线程数，也就是一个block有8个wrap，每个block要计算的元素个数是256，一个warp有32个线程


__device__ void warpReduce(volatile float *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_kernel_v4(float *g_idata, float *g_odata) {
  // 申请共享内存
  __shared__ float sdata[BLOCK_SIZE];

  // 每个线程从全局内存中读取一个数据到共享内存
  unsigned int tid = threadIdx.x;   // 线程id
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;  // 全局id
  // sdata[tid] = g_idata[i];  // 从全局内存中读取数据到共享内存
  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];  // 从全局内存中读取数据到共享内存 一个线程读取两个数据

  __syncthreads();   // 同步

  // 在共享内存上做reduce
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();  // 同步
  }
  if (tid < 32) warpReduce(sdata, tid);

  // 写结果到全局内存
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

int main() {
  float *input_host = (float *)malloc(N * sizeof(float));
  float *input_device;
  cudaMalloc((void **)&input_device, N * sizeof(float));  //  申请显存
  for (int i = 0; i < N; i++) {
    input_host[i] = 2.0;
  }
  cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);  // 从主机内存拷贝到显存

  int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
  float *output_host = (float *)malloc((N / BLOCK_SIZE) * sizeof(float));
  float *output_device;
  cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));  //  申请显存

  dim3 grid(N / BLOCK_SIZE, 1);   // 会自动在最后边补1  grid(N/BLOCK_SIZE, 1, 1)
  dim3 block(BLOCK_SIZE, 1);
  reduce_kernel_v4<<<grid, block>>>(input_device, output_device);
  cudaMemcpy(output_device, output_host, block_num * sizeof(float), cudaMemcpyDeviceToHost);  // 从显存拷贝到主机内存
  return 0;
}
