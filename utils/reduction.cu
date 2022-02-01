#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include "types.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmemUnroll(u32 *g_idata, u32 *g_odata, u64 n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    u32 *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx + 3 * blockDim.x < n) {
        u32 a1 = g_idata[idx];
        u32 a2 = g_idata[idx + blockDim.x];
        u32 a3 = g_idata[idx + 2 * blockDim.x];
        u32 a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32) {
        volatile u32 *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
//        printf("%d, %d\n", blockIdx.x, idata[0]);
    }
}

__global__ void reduceGmemBatchSum(u16 *g_idata, u16 *g_odata, u64 n, u64 n_batches, u32 batch_size) {
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread sums batch_size elements
    u64 sum = 0;
    for (u32 i = 0; i < batch_size; i++) {
        sum += g_idata[idx * batch_size + i];
    }

    g_odata[idx] = sum / batch_size;
}

__global__ void reduceGmemBatchMin(u16 *g_idata, u16 *g_odata, u64 n, u64 n_batches, u32 batch_size) {
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    u16 min = g_idata[idx * batch_size];
    for (u32 i = 1; i < batch_size; i++) {
        u16 val = g_idata[idx * batch_size + i];
        if (val < min) min = val;
    }

    g_odata[idx + n_batches] = min;
}


__global__ void reduceGmemBatchMax(u16 *g_idata, u16 *g_odata, u64 n, u64 n_batches, u32 batch_size) {
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    u16 max = g_idata[idx * batch_size];
    for (u32 i = 1; i < batch_size; i++) {
        u16 val = g_idata[idx * batch_size + i];
        if (val > max) max = val;
    }

    g_odata[idx + n_batches * 2] = max;
}

#endif