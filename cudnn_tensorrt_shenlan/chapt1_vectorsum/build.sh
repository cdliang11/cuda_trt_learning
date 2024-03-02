# CPU
g++ -O3 main_cpu.cpp -o VectorSumCPU

# GPU
/usr/local/cuda/bin/nvcc main_gpu.cu -o VectorSumGPU

