#include <iostream>
#include "cpu.cu"
#include "utils/time.cu"
#include "utils/warmup.cu"
#include "utils/stats.cu"

#define BITS 12
#define TABLE_SIZE 4096
#define MEMORY_POWER 31
#define BATCH_SIZE 1024

__global__ void simple_gpu(u16 *res) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) {
        u64 a = i;
        u16 c = 0;
        while (a != 1) {
            if (a % 2 == 0) {
                a = a / 2;
                c++;
            } else {
                a = (3 * a + 1) / 2;
                c += 2;
            }
        }
        res[i] = c;
    } else {
        res[i] = 0;
    }
}

__global__ void
gpu_LUT(u16 *res, u64 *gmem_table_BC, u16 *gmem_table_D, u16 *table_E, u64 memory_size, u64 offset, u64 n_batches) {
    __shared__ u64 table_BC[TABLE_SIZE];
    __shared__ u16 table_D[TABLE_SIZE];

    if (TABLE_SIZE > BATCH_SIZE) {
        int iters = TABLE_SIZE / BATCH_SIZE;
        for (int i = 0; i < iters; i++) {
            int idx = i * BATCH_SIZE + threadIdx.x;
            if (idx < TABLE_SIZE) {
                table_BC[idx] = gmem_table_BC[idx];
                table_D[idx] = gmem_table_D[idx];
            }
        }
    } else if (TABLE_SIZE == BATCH_SIZE) {
        table_BC[threadIdx.x] = gmem_table_BC[threadIdx.x];
        table_D[threadIdx.x] = gmem_table_D[threadIdx.x];
    } else {
        int idx = threadIdx.x;
        if (idx < TABLE_SIZE) {
            table_BC[idx] = gmem_table_BC[idx];
            table_D[idx] = gmem_table_D[idx];
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
        u64 num = i;
        u16 count = 0;

        while (num >= memory_size) {
            u64 n_high = num >> BITS;
            u64 n_low = num - (n_high << BITS);
            u64 bc = table_BC[n_low];
            u32 b = (bc >> 16) >> 16;
            u32 c = bc - ((((u64) b) << 16) << 16);
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

void generate_LUT(u64 *BC_table, u16 *D_table) {
    u32 A = TABLE_SIZE;
    for (u32 i = A; i < 2 * A; i++) {
        u32 n_h = i >> BITS;
        u32 n_l = i - (n_h << BITS);
        u64 b = A;
        u64 c = n_l;
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
                u64 BC = ((b << 16) << 16) + c;
                BC_table[n_l] = BC;
                D_table[n_l] = d;
                break;
            }
        }
    }
}

int main() {
    bool verify = false;
    int n_tests = 10;
    u64 memory_size = power(MEMORY_POWER);
    u64 N_to_calc = power(34);
    u64 offset = power(40);
    u64 n_batches = N_to_calc / BATCH_SIZE;
    u64 n_threads = n_batches;
    u64 block_size = 1024;
    u64 grid_size = n_threads / block_size;

    // print parameters
    std::cout << "N_to_calc: " << N_to_calc << std::endl;
    std::cout << "BITS: " << BITS << " TABLE_SIZE: " << TABLE_SIZE << std::endl;
    std::cout << "MEMORY_POWER: " << MEMORY_POWER << std::endl;
    std::cout << "n_batches: " << n_batches << " n_threads: " << n_threads << " block_size: " << block_size
              << " grid_size: " << grid_size << std::endl;

    double *gpu_time = new double[n_tests];
    double *gpu_alloc_time = new double[n_tests];
    double *gpu_copy_time = new double[n_tests];

    // calculate result for cpu
    std::cout << "CPU started" << std::endl;
    u64 *table_BC_cpu = (u64 *) malloc(TABLE_SIZE * sizeof(u64));
    u16 *table_D_cpu = (u16 *) malloc(TABLE_SIZE * sizeof(u16));
    generate_LUT(table_BC_cpu, table_D_cpu);
    std::cout << "LUT generation finished" << std::endl;

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
        // allocate gpu memory for tables
        u64 *table_BC_gpu;
        u16 *table_D_gpu;
        u16 *table_E_gpu;
        cudaMalloc((void **) &table_BC_gpu, TABLE_SIZE * sizeof(u64));
        cudaMalloc((void **) &table_D_gpu, TABLE_SIZE * sizeof(u16));
        cudaMalloc((void **) &table_E_gpu, memory_size * sizeof(u16));
        // allocate gpu memory for result
        u16 *gpu_res;
        cudaMalloc(&gpu_res, n_batches * 3 * sizeof(u16));
        // copy tables BC, D and calculate table E
        cudaMemcpy(table_BC_gpu, table_BC_cpu, TABLE_SIZE * sizeof(u64), cudaMemcpyHostToDevice);
        cudaMemcpy(table_D_gpu, table_D_cpu, TABLE_SIZE * sizeof(u16), cudaMemcpyHostToDevice);
        simple_gpu<<<memory_size / block_size, block_size>>>(table_E_gpu);
        cudaDeviceSynchronize();
        end = getSecond();
        gpu_alloc_time[t] = end - start;

        // GPU work
        start = getSecond();
        gpu_LUT<<<grid_size, block_size>>>(gpu_res, table_BC_gpu, table_D_gpu, table_E_gpu,
                                           memory_size, offset, n_batches);
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
        cudaFree(table_BC_gpu);
        cudaFree(table_D_gpu);
        cudaFree(table_E_gpu);

        if (!success) {
            std::cout << "Test " << t << " failed" << std::endl;
            return 1;
        }
        std::cout << "Test " << t << " completed" << std::endl;
    }

    return print_stats_gpu(gpu_time, gpu_alloc_time, gpu_copy_time, n_tests);
}