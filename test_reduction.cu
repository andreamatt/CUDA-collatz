#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"
#include "utils/reduction.cu"
#include "gpu.cu"


int main() {
    int n_tests = 1;
    double *cpu_time = new double[n_tests];
    double *cpu_alloc_time = new double[n_tests];
    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    for (int t = 0; t < n_tests; t++) {
        u64 N = 1 << 29;
        double start, end;

#pragma region CPU
        start = getSecond();
        u16 *res1 = (u16 *) malloc(N * sizeof(u16));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        res1[0] = 0;
        dynamic_cpu(res1, N);
        u64 sum_cpu = 0;
        for (u64 i = 0; i < N; i++) {
            sum_cpu += res1[i];
        }
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU
        warmup();

        int block_size = 1024;
        int grid_size = N / block_size;

        start = getSecond();
        u32 *gpu_res1;
        cudaMalloc(&gpu_res1, N * sizeof(u32));
        u32 *gpu_res2;
        cudaMalloc(&gpu_res2, grid_size * sizeof(u32));
        u32 *gpu_cpu = (u32 *) malloc(grid_size * sizeof(u32));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        simple_gpu32<<<grid_size, block_size>>>(gpu_res1, N);
        cudaDeviceSynchronize();
        std::cout << "GPU simple" << std::endl;
        // reduce
        reduceGmemUnroll<<<grid_size / 4, block_size>>>(gpu_res1, gpu_res2, N);
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
        cudaMemcpy(gpu_cpu, gpu_res2, grid_size / 4 * sizeof(u32), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        u64 sum_gpu = 0;
        std::cout << "GPU reduce" << std::endl;
        for (u64 i = 0; i < grid_size / 4; i++) {
//            std::cout << gpu_cpu[i] << std::endl;
            sum_gpu += gpu_cpu[i];
        }
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // COMPARISON

        // FREE memory
        free(res1);
        free(gpu_cpu);
        cudaFree(gpu_res1);
        cudaFree(gpu_res2);

        // PRINT RESULTS
        std::cout << "CPU sum: " << sum_cpu << std::endl;
        std::cout << "GPU sum: " << sum_gpu << std::endl;
    }

    return print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}