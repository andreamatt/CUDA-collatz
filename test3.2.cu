#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"
#include "gpu.cu"


int main() {
    int n_tests = 1;
    double *cpu_time = new double[n_tests];
    double *cpu_alloc_time = new double[n_tests];
    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    for (int t = 0; t < n_tests; t++) {
        u64 N = 1 << 25;
        u32 batch_size = 1 << 10;
        u64 N_batches = N / batch_size;
        u64 arr_size = 3 * N_batches;
        double start, end;

#pragma region CPU
        start = getSecond();
        u16 *cpu_res1 = (u16 *) malloc(arr_size * sizeof(u16));
        u16 *cpu_res2 = (u16 *) malloc(arr_size * sizeof(u16));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        simple_cpu_batch(cpu_res1, N, N_batches, batch_size);
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU
        warmup();

        u64 work_batch_size = batch_size * 1024;
        u64 work_N_batches = N_batches / work_batch_size;
        int block_size = 1024;
        int grid_size = (N + block_size - 1) / (block_size * batch_size);
        std::cout << "Grid size: " << grid_size << std::endl;
        std::cout << "Block size: " << block_size << std::endl;

        start = getSecond();
        u16 *gpu_res;
        cudaMalloc(&gpu_res, arr_size * sizeof(u16));
        u16 *gpu_work_res;
        cudaMalloc(&gpu_work_res, work_batch_size * sizeof(u16));
        u16 *gpu_work_redu;
        cudaMalloc(&gpu_work_redu, work_batch_size * sizeof(u16) / 2);
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        for (u64 w = 0; w < work_N_batches; w++) {
            // 1. calculate batch
            u64 work_start = w * work_batch_size;
            u64 work_end = work_start + work_batch_size;
            simple_gpu_offset<<<grid_size, block_size>>>(gpu_work_res, work_start, work_end);
            cudaDeviceSynchronize();

            // 2. reduce batch
        }
        simple_gpu_batch_1<<<grid_size, block_size>>>(gpu_res, N, N_batches, batch_size);
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
        cudaMemcpy(cpu_res2, gpu_res, arr_size * sizeof(u16), cudaMemcpyDeviceToHost);
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // COMPARISON
        bool success = compare_arrays(cpu_res1, cpu_res2, arr_size);

        // FREE memory
        free(cpu_res1);
        free(cpu_res2);
        cudaFree(gpu_res);

        if (!success) {
            std::cout << "Test " << t << " failed" << std::endl;
            return 1;
        }
        std::cout << "Test " << t << " completed" << std::endl;
    }

    return print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}