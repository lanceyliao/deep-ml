#include <cuda_runtime.h>

// 线程块大小（建议定义为 2 的幂次以保证树状归约对齐）
#define BLOCK_SIZE 256

__global__ void reduce_kernel(const float* in, float* out, int N) {
    // 1. 静态分配共享内存：仅在当前 Block 内所有 Thread 共享
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 数据加载与边界越界保护
    // 越界位置填充加法单位元 0.0f，保证归约循环中无需分支判断
    sdata[tid] = (global_idx < N) ? in[global_idx] : 0.0f;

    // 块内同步：确保所有线程完成 Global -> Shared Memory 数据拷贝
    __syncthreads();

    // 3. 共享内存树状折叠归约 (Tree Reduction)
    // 步长 stride 初始为块大小的一半，每轮减半，直到 stride == 0
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // 关键屏障：保证上一轮所有的 sdata[tid + s] 读取/写入都已完成
        __syncthreads();
    }

    // 4. 跨 Block 累加：每个 Block 的 0 号线程持有所属 Block 的局部和
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        *output = 0.0f;
        return;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;

    size_t bytes_in = N * sizeof(float);
    size_t bytes_out = sizeof(float);

    // 1. 显存分配
    cudaMalloc(&d_in, bytes_in);
    cudaMalloc(&d_out, bytes_out);

    // 2. 数据传输与初始化
    cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice);
    // 累加器初始化为 0.0f
    cudaMemset(d_out, 0, bytes_out);

    // 3. 计算 Grid / Block 布局
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 4. 启动 Kernel
    reduce_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);

    // 5. 将最终标量结果拷回 Host
    cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost);

    // 6. 释放 Device 资源
    cudaFree(d_in);
    cudaFree(d_out);
}