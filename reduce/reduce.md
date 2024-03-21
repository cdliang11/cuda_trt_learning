# reduce 优化入门

> 对cuda官方博客的学习和总结

## 背景

reduce操作是对一个数组求sum、min、max、avg等。reduce又称为规约，即递归约减，最终输出相比于输入一般在维度上会递减。

以reduce_sum问题为例，一个长度为8的数组求和之后得到的输出只有一个数，从1维数组变成一个标量。

## 硬件环境
NVIDIA GTX1650 峰值带宽，CUDA版本为11.8

## baseline

以树形图的方式去执行数据累加，最终得到总和。但是由于GPU没有对global memory的同步操作，所以分成多个阶段的方式来避免global memory的操作。如下：


算法实现：
```c++
__global__ void reduce_kernel_v0(float *g_idata, float *g_odata) {
  // 申请共享内存
  __shared__ float sdata[BLOCK_SIZE];

  // 每个线程从全局内存中读取一个数据到共享内存
  unsigned int tid = threadIdx.x;   // 线程id
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局id
  sdata[tid] = g_idata[i];  // 从全局内存中读取数据到共享内存

  __syncthreads();   // 同步， 这部分是一个潜在耗时的点

  // 在共享内存上做reduce
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];  // 每次将 (2k+1)s 加到 2ks上
    }
    __syncthreads();  // 同步
  }

  // 写结果到全局内存
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}
```

g_idata表示输入数据的指针，而g_odata表示输出数据的指针。然后把global memory数据加载到shard memory中，接着在shared memory中对数据进行reduce_sum操作，最后将结果写入global memory中。

编译，性能如下：


但baseline有明显的低效之处。
```c++
// 在共享内存上做reduce
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // 问题1：存在warp divergent
    // 问题2: %取模操作很慢
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();  // 同步
  }
```
有两个问题：
- 问题1：存在warp divergent
- 问题2: %取模操作很慢

> 什么是warp divergent？
> 在CUDA编程中，wrap divergent指的是在一个wrap中不同线程之间执行不同的代码路径(即分支语句的不同分支)，导致warp中的线程不能同时执行相同的指令，从而影响性能。
>
> 但一个warp中的线程执行分支语句，如果不是所有的线程都选择同一个分支路径，就会发生warp divergent。因为warp中的线程是以SIMD(Single Instruction Multiple Data)方式执行，每个线程必须执行相同的指令。如果不同的线程选择了不同的分支路径，就会出现某些线程被阻塞等待其他线程完成它们的操作，这就浪费了处理器的时间。


## 优化1

```c++
__global__ void reduce_kernel_v1(float *g_idata, float *g_odata) {
  // 申请共享内存
  __shared__ float sdata[BLOCK_SIZE];

  // 每个线程从全局内存中读取一个数据到共享内存
  unsigned int tid = threadIdx.x;   // 线程id
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局id
  sdata[tid] = g_idata[i];  // 从全局内存中读取数据到共享内存

  __syncthreads();   // 同步， 这部分是一个潜在耗时的点

  // 在共享内存上做reduce
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // 交错寻址
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();  // 同步
  }

  // 写结果到全局内存
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}
```

针对baseline中的warp divergent问题进行优化。通过调整baseline中的分支判断代码，使得更多的线程可以走到同一个分支里边，从而降低资源浪费。

具体做法：把`if (tid % (2*s) == 0)` 替换成strided index的方式，即`int index`






