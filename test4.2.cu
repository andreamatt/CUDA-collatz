#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"

#define BITS 9
#define TABLE_SIZE 512
#define BATCH_SIZE 1024

__device__ __constant__ u32 table_B[TABLE_SIZE];
__device__ __constant__ u32 table_C[TABLE_SIZE];
__device__ __constant__ u16 table_D[TABLE_SIZE];
__device__ __constant__ u16 table_E[TABLE_SIZE];


__global__ void gpu_LUT(u16 *res, u64 offset, u64 n_batches) {
    u64 id = blockIdx.x * blockDim.x + threadIdx.x;
    u64 i_start = BATCH_SIZE * id + offset;
    u64 i_end = i_start + BATCH_SIZE;
    u16 min_c = UINT16_MAX;
    u16 max_c = 0;
    u32 sum_c = 0;
    if (i_start == 0) i_start = 1;
    for (u64 i = i_start; i < i_end; i++) {
        u64 num = i;
        u16 count = 0;

        while (num >= TABLE_SIZE) {
            u64 n_high = num >> BITS;
            u64 n_low = num - (n_high << BITS);
            u32 b = table_B[n_low];
            u32 c = table_C[n_low];
            count += table_D[n_low];
            num = n_high * b + c;
        }
        // end of cycle, use LUT for remaining
        count += table_E[num];

        // update stats
        if (count < min_c) {
            min_c = count;
        }
        if (count > max_c) {
            max_c = count;
        }
        sum_c += count;
    }
    res[id] = sum_c / BATCH_SIZE;
    res[id + n_batches] = min_c;
    res[id + n_batches * 2] = max_c;
}

void generate_LUT(u32 *B_table, u32 *C_table, u16 *D_table) {
    u32 A = TABLE_SIZE;
    for (u32 i = A; i < 2 * A; i++) {
        u32 n_h = i >> BITS;
        u32 n_l = i - (n_h << BITS);
        u32 b = A;
        u32 c = n_l;
        u16 d = 0;
        while (true) {
            if (b % 2 == 0) {
                if (c % 2 == 0) {
                    b = b / 2;
                    c = c / 2;
                } else {
                    b = b * 3;
                    c = c * 3 + 1;
                }
                d++;
            } else {
                B_table[n_l] = b;
                C_table[n_l] = c;
                D_table[n_l] = d;
                break;
            }
        }
    }
}

int main() {
    bool verify = false;
    int n_tests = 1;
    u64 N_to_calc = power(30);
    u64 offset = power(40);
    u64 n_batches = N_to_calc / BATCH_SIZE;
    u64 n_threads = n_batches;
    u64 block_size = 1024;
    u64 grid_size = n_threads / block_size;

    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    // defines the three tables
    u32 *table_B_cpu = (u32 *) malloc(TABLE_SIZE * sizeof(u32));
    u32 *table_C_cpu = (u32 *) malloc(TABLE_SIZE * sizeof(u32));
    u16 *table_D_cpu = (u16 *) malloc(TABLE_SIZE * sizeof(u16));
    u16 *table_E_cpu = (u16 *) malloc(TABLE_SIZE * sizeof(u16));
    // calculate the three tables
    std::cout << "CPU started" << std::endl;
    generate_LUT(table_B_cpu, table_C_cpu, table_D_cpu);
    std::cout << "LUT generation finished" << std::endl;
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
        // copy all 4 tables to gpu constant memory
        cudaMemcpyToSymbol(table_B, table_B_cpu, TABLE_SIZE * sizeof(u32), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(table_C, table_C_cpu, TABLE_SIZE * sizeof(u32), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(table_D, table_D_cpu, TABLE_SIZE * sizeof(u16), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(table_E, table_E_cpu, TABLE_SIZE * sizeof(u16), 0, cudaMemcpyHostToDevice);
        // allocate gpu memory for result
        u16 *gpu_res;
        cudaMalloc(&gpu_res, n_batches * 3 * sizeof(u16));
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        gpu_LUT<<<grid_size, block_size>>>(gpu_res, offset, n_batches);
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

        if (!success) {
            std::cout << "Test " << t << " failed" << std::endl;
            return 1;
        }
        std::cout << "Test " << t << " completed" << std::endl;
    }

    return print_stats_gpu(gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}