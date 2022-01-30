#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "gpu.cu"

__global__ void simple_gpu_batch(u16 *res, u64 batch_idx, u64 batch_size){
    u64 i = blockIdx.x * blockDim.x + threadIdx.x +1;
    if(i < batch_size){
        u64 a = i + batch_idx;
        u16 c = 0;
        while (a != 1) {
            if (a % 2 == 0) {
                a = a / 2;
                c++;
            } else {
                a = (3 * a + 1) / 2;
                c+=2;
            }
        }
        res[i] = c;
    }
}

int main() {
    int exp = 34;
    u64 N = 1;
    for (int i = 0; i < exp; i++) {
        N *= 2;
    }

    u64 batch_size = 1 << 30;
    u64 num_batches = N / batch_size;
    double start, end;

    u16 *cpu_res = (u16 *) malloc(sizeof(u16) * batch_size);

    // GPU PART
    start = getSecond();
    warmup();
    end = getSecond();
    std::cout << "Warmup time: " << end - start << std::endl;

    start = getSecond();
    u16 *res3;
    cudaMalloc(&res3, batch_size * sizeof(u16));
    end = getSecond();
    std::cout << "GPU allocation time: " << end - start << std::endl;

//    start = getSecond();
//    int block_size = 1024;
//    u64 grid_size = (N + block_size - 1) / block_size;
//    std::cout << "Grid size: " << grid_size << std::endl;
//    std::cout << "Block size: " << block_size << std::endl;
//    simple_gpu<<<grid_size, block_size>>>(res3, N);
//    cudaDeviceSynchronize();
//    end = getSecond();
//    std::cout << "GPU total time: " << end - start << std::endl;

    start = getSecond();
    int block_size = 1024;
    u64 grid_size = batch_size / block_size;
    std::cout << "Grid size: " << grid_size << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    for (u64 i = 0; i < num_batches; i++) {
        simple_gpu_batch<<<grid_size, block_size>>>(res3, i, batch_size);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_res, res3, batch_size * sizeof(u16), cudaMemcpyDeviceToHost);

        auto code = cudaGetLastError();
        if(code != cudaSuccess)
        {
            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
            return 1;
        }
    }
    end = getSecond();
    std::cout << "GPU total time: " << end - start << std::endl;

    free(cpu_res);
    cudaFree(res3);

    return 0;
}