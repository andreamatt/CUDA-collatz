#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"
#include "utils/reduction.cu"
#include "gpu.cu"


int main() {
    for (int b_p = 10; b_p <= 10; b_p++) {
        int n_tests = 5;
        u64 N = 1 << 30;
        u32 batch_size = 1 << b_p;
        u64 N_batches = N / batch_size;
        double *cpu_time = new double[n_tests];
        double *cpu_alloc_time = new double[n_tests];
        double *gpu_time = new double[n_tests];
        double *gpu_alloc_time = new double[n_tests];
        double *gpu_copy_time = new double[n_tests];

        for (int t = 0; t < n_tests; t++) {
            u64 arr_size = 3 * N_batches;
            double start, end;

#pragma region CPU
            start = getSecond();
            u16 *cpu_res1 = (u16 *) malloc(arr_size * sizeof(u16));
            u16 *cpu_res2 = (u16 *) malloc(arr_size * sizeof(u16));
            end = getSecond();
            cpu_alloc_time[t] = end - start;

            start = getSecond();
//        simple_cpu_batch(cpu_res1, N, N_batches, batch_size);
            end = getSecond();
            cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU
            warmup();

            int block_size = 1024;
            int grid_size = N / block_size;
            int reduction_grid_size = N_batches;
            int reduction_block_size = 1024;
//        std::cout << "Grid size: " << grid_size << std::endl;
//        std::cout << "Block size: " << block_size << std::endl;
//        std::cout << "Reduction grid size: " << reduction_grid_size << std::endl;
//        std::cout << "N_batches: " << N_batches << std::endl;

            start = getSecond();
            u32 *gpu_res;
            cudaMalloc(&gpu_res, N * sizeof(u32));
            u16 *gpu_redu;
            cudaMalloc(&gpu_redu, arr_size * sizeof(u16));
            end = getSecond();
            gpu_alloc_time[t] = end - start;

            // GPU work
            start = getSecond();
            simple_gpu32<<<grid_size, block_size>>>(gpu_res, N);
            reduceGmemUnrollSum<<<reduction_grid_size / 4, reduction_block_size>>>(gpu_res, gpu_redu, N);
            cudaDeviceSynchronize();
            end = getSecond();
            gpu_time[t] = end - start;


            // CHECK CUDA ERRORS
//        auto code = cudaGetLastError();
//        if (code != cudaSuccess) {
//            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
//        }

            // COPY BACK
            start = getSecond();
            cudaMemcpy(cpu_res2, gpu_redu, arr_size * sizeof(u16), cudaMemcpyDeviceToHost);
            end = getSecond();
            gpu_copy_time[t] = end - start;
#pragma endregion

            // COMPARISON
//        bool success = compare_arrays(cpu_res1, cpu_res2, arr_size);

            // FREE memory
            free(cpu_res1);
            free(cpu_res2);
            cudaFree(gpu_res);

//        if (!success) {
//            std::cout << "Test " << t << " failed" << std::endl;
//            return 1;
//        }
            std::cout << "Test " << t << " completed" << std::endl;
        }
        std::cout << "Batch size: " << batch_size << std::endl;
        print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
        std::cout << std::endl;
    }

    return 0;
}