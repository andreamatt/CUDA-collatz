#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"
#include "gpu.cu"

__global__ void copy1632(u16 *res16, u32 *res32, u64 n) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res32[i] = res16[i];
    }
}

int main() {
    int n_tests = 50;

    u64 N = 1 << 29;
    int block_size = 1024;
    int grid_size = N / block_size;

    double *cpu_time = new double[n_tests];
    double *cpu_alloc_time = new double[n_tests];
    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    warmup();

#pragma region INT16
    for (int t = 0; t < n_tests; t++) {
        double start, end;

#pragma region CPU
        start = getSecond();
        u16 *res = (u16 *) malloc(N * sizeof(u16));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU

        start = getSecond();
        u16 *gpu_res;
        cudaMalloc(&gpu_res, N * sizeof(u16));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        simple_gpu<<<grid_size, block_size>>>(gpu_res);
        cudaDeviceSynchronize();
        end = getSecond();
        gpu_time[t] = end - start;


        // CHECK CUDA ERRORS
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
        }

        // COPY BACK
        start = getSecond();
        cudaMemcpy(res, gpu_res, N * sizeof(u16), cudaMemcpyDeviceToHost);
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // FREE memory
        free(res);
        cudaFree(gpu_res);
    }
#pragma endregion

    print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);

#pragma region INT32
    for (int t = 0; t < n_tests; t++) {
        double start, end;

#pragma region CPU
        start = getSecond();
        u32 *res = (u32 *) malloc(N * sizeof(u32));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU

        start = getSecond();
        u32 *gpu_res;
        cudaMalloc(&gpu_res, N * sizeof(u32));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        simple_gpu32<<<grid_size, block_size>>>(gpu_res, N);
        cudaDeviceSynchronize();
        end = getSecond();
        gpu_time[t] = end - start;


        // CHECK CUDA ERRORS
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
        }

        // COPY BACK
        start = getSecond();
        cudaMemcpy(res, gpu_res, N * sizeof(u32), cudaMemcpyDeviceToHost);
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // FREE memory
        free(res);
        cudaFree(gpu_res);
    }
#pragma endregion

    print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);

#pragma region INT1632
    for (int t = 0; t < n_tests; t++) {
        double start, end;

#pragma region CPU
        start = getSecond();
        u32 *res32 = (u32 *) malloc(N * sizeof(u32));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU

        start = getSecond();
        u16 *gpu_res16;
        cudaMalloc(&gpu_res16, N * sizeof(u16));
        u32 *gpu_res32;
        cudaMalloc(&gpu_res32, N * sizeof(u32));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        simple_gpu<<<grid_size, block_size>>>(gpu_res16);
        copy1632<<<grid_size, block_size>>>(gpu_res16, gpu_res32, N);
        cudaDeviceSynchronize();
        end = getSecond();
        gpu_time[t] = end - start;


        // CHECK CUDA ERRORS
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cout << "Error CP: " << cudaGetErrorString(code) << std::endl;
        }

        // COPY BACK
        start = getSecond();
        cudaMemcpy(res32, gpu_res32, N * sizeof(u32), cudaMemcpyDeviceToHost);
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // FREE memory
        free(res32);
        cudaFree(gpu_res16);
        cudaFree(gpu_res32);
    }
#pragma endregion


    print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);

    return 0;
}