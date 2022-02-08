#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"

#define BITS 14
#define TABLE_SIZE 16384
#define BATCH_SIZE 1024

__global__ void gpu_LUT_E(u16 *res, u16 *gmem_table_E, u64 offset, u64 n_batches) {
    __shared__ u16 table_E[TABLE_SIZE];

    if (TABLE_SIZE > BATCH_SIZE) {
        int iters = TABLE_SIZE / BATCH_SIZE;
        for (int i = 0; i < iters; i++) {
            int idx = i * BATCH_SIZE + threadIdx.x;
            if (idx < TABLE_SIZE) {
                table_E[idx] = gmem_table_E[idx];
            }
        }
    } else if (TABLE_SIZE == BATCH_SIZE) {
        table_E[threadIdx.x] = gmem_table_E[threadIdx.x];
    } else {
        int idx = threadIdx.x;
        if (idx < TABLE_SIZE) {
            table_E[idx] = gmem_table_E[idx];
        }
    }
    __syncthreads();

    u64 id = blockIdx.x * blockDim.x + threadIdx.x;
    u64 i_start = BATCH_SIZE * id + offset;
    u64 i_end = i_start + BATCH_SIZE;
    u16 min_c = UINT16_MAX;
    u16 max_c = 0;
    u32 sum_c = 0;
    if (i_start == 0) i_start = 1;
    for (u64 i = i_start; i < i_end; i++) {
        u64 a = i;
        u16 c = 0;
        while (a >= TABLE_SIZE) {
            if (a % 2 == 0) {
                a = a / 2;
                c++;
            } else {
                a = (3 * a + 1) / 2;
                c += 2;
            }
        }
        // LUT with table_E
        c += table_E[a];

        if (c < min_c) {
            min_c = c;
        }
        if (c > max_c) {
            max_c = c;
        }
        sum_c += c;
    }
    res[id] = sum_c / BATCH_SIZE;
    res[id + n_batches] = min_c;
    res[id + n_batches * 2] = max_c;
}

int main() {
    bool verify = false;
    int n_tests = 10;
    u64 N_to_calc = power(34);
    u64 offset = power(40);
    u64 n_batches = N_to_calc / BATCH_SIZE;
    u64 n_threads = n_batches;
    u64 block_size = BATCH_SIZE;
    u64 grid_size = n_threads / block_size;

    // print parameters
    std::cout << "N_to_calc: " << N_to_calc << std::endl;
    std::cout << "BITS: " << BITS << " TABLE_SIZE: " << TABLE_SIZE << std::endl;
    std::cout << "n_batches: " << n_batches << " n_threads: " << n_threads << " block_size: " << block_size
              << " grid_size: " << grid_size << std::endl;

    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    // defines the three tables
    u16 *table_E_cpu = (u16 *) malloc(TABLE_SIZE * sizeof(u16));
    // calculate the three tables
    std::cout << "CPU started" << std::endl;
    dynamic_cpu(table_E_cpu, TABLE_SIZE);
    std::cout << "Table E finished" << std::endl;

    // calculate result for cpu
    std::cout << "CPU started" << std::endl;
    u16 *cpu_res = (u16 *) malloc(n_batches * 3 * sizeof(u16));
    u16 *cpu_res_compare = (u16 *) malloc(n_batches * 3 * sizeof(u16));
    if (verify) {
        simple_cpu_batch(cpu_res, offset, n_batches, BATCH_SIZE);
    }
    std::cout << "CPU finished" << std::endl;

    warmup();

    for (int t = 0; t < n_tests; t++) {
        double start, end;

        start = getSecond();
        // copy table to gpu memory
        u16 *gmem_table_E;
        cudaMalloc(&gmem_table_E, TABLE_SIZE * sizeof(u16));
        cudaMemcpy(gmem_table_E, table_E_cpu, TABLE_SIZE * sizeof(u16), cudaMemcpyHostToDevice);
        // allocate gpu memory for result
        u16 *gpu_res;
        cudaMalloc(&gpu_res, n_batches * 3 * sizeof(u16));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        gpu_LUT_E<<<grid_size, block_size>>>(gpu_res, gmem_table_E, offset, n_batches);
        cudaDeviceSynchronize();
        end = getSecond();
        gpu_time[t] = end - start;


        // CHECK CUDA ERRORS
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
        }

        // COMPARISON
        bool success = true;
        if (verify) {
            // COPY BACK
            start = getSecond();
            cudaMemcpy(cpu_res_compare, gpu_res, n_batches * 3 * sizeof(u16), cudaMemcpyDeviceToHost);
            end = getSecond();
            gpu_copy_time[t] = end - start;
            success = compare_arrays(cpu_res, cpu_res_compare, n_batches * 3);
        } else {
            gpu_copy_time[t] = 0;
        }

        // FREE memory
        cudaFree(gpu_res);
        cudaFree(gmem_table_E);

        if (!success) {
            std::cout << "Test " << t << " failed" << std::endl;
            return 1;
        }
        std::cout << "Test " << t << " completed" << std::endl;
    }

    return print_stats_gpu(gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}