#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"
#include "gpu.cu"


int main() {
    int n_tests = 10;
    double *cpu_time = new double[n_tests];
    double *cpu_alloc_time = new double[n_tests];
    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    for (int t = 0; t < n_tests; t++) {
        u64 N = 1 << 30;
        double start, end;

#pragma region CPU
        start = getSecond();
        u16 *res1 = (u16 *) malloc(N * sizeof(u16));
        u16 *res2 = (u16 *) malloc(N * sizeof(u16));
        end = getSecond();
        cpu_alloc_time[t] = end - start;

        start = getSecond();
        dynamic_cpu(res1, N);
        end = getSecond();
        cpu_time[t] = end - start;
#pragma endregion

#pragma region GPU
        warmup();

        start = getSecond();
        u16 *gpu_res1;
        cudaMalloc(&gpu_res1, N * sizeof(u16));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        int block_size = 1024;
        int grid_size = (N + block_size - 1) / block_size;
//        std::cout << "Grid size: " << grid_size << std::endl;
//        std::cout << "Block size: " << block_size << std::endl;
        simple_gpu<<<grid_size, block_size>>>(gpu_res1);
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
        cudaMemcpy(res2, gpu_res1, N * sizeof(u16), cudaMemcpyDeviceToHost);
        res2[0] = 0;
        end = getSecond();
        gpu_copy_time[t] = end - start;
#pragma endregion

        // COMPARISON
        bool success = compare_arrays(res1, res2, N);

        // FREE memory
        free(res1);
        free(res2);
        cudaFree(gpu_res1);

        if (!success) {
            std::cout << "Test " << t << " failed" << std::endl;
            return 1;
        }
        std::cout << "Test " << t << " completed" << std::endl;
    }

    return print_stats(cpu_time, cpu_alloc_time, gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}