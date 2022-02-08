#include <iostream>
#include <fstream>
#include "utils/time.cu"
#include "cpu.cu"

#define BITS 12
#define TABLE_SIZE 4096
#define MEMORY_POWER 31
#define BATCH_SIZE 1024
#define JOB_BATCH_POWER 36

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
gpu_LUT(u16 *res_avg, u16 *res_min, u16 *res_max, u64 *gmem_table_BC, u16 *gmem_table_D, u16 *table_E, u64 memory_size,
        u64 offset, u64 n_batches) {
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
    res_avg[id] = sum_c / BATCH_SIZE;
    res_min[id] = min_c;
    res_max[id] = max_c;
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
    double start, end;
    start = getSecond();

    u64 N_to_calc = power(36);
    u64 offset = 0;
    u64 job_batch_size = power(JOB_BATCH_POWER);
    u64 n_jobs = N_to_calc / job_batch_size;
    u64 tot_batches = N_to_calc / BATCH_SIZE;

    u64 memory_size = power(MEMORY_POWER);
    u64 n_batches_per_job = job_batch_size / BATCH_SIZE;
    u64 n_threads = n_batches_per_job;
    u64 block_size = BATCH_SIZE;
    u64 grid_size = n_threads / block_size;

    // calculate result for cpu
    std::cout << "CPU started" << std::endl;
    u64 *table_BC_cpu = (u64 *) malloc(TABLE_SIZE * sizeof(u64));
    u16 *table_D_cpu = (u16 *) malloc(TABLE_SIZE * sizeof(u16));
    generate_LUT(table_BC_cpu, table_D_cpu);
    std::cout << "LUT generation finished" << std::endl;
    u16 *cpu_res_avg = (u16 *) malloc(tot_batches * sizeof(u16));
    u16 *cpu_res_min = (u16 *) malloc(tot_batches * sizeof(u16));
    u16 *cpu_res_max = (u16 *) malloc(tot_batches * sizeof(u16));
    std::cout << "CPU finished" << std::endl;

    // allocate gpu memory for tables
    u64 *table_BC_gpu;
    u16 *table_D_gpu;
    u16 *table_E_gpu;
    cudaMalloc((void **) &table_BC_gpu, TABLE_SIZE * sizeof(u64));
    cudaMalloc((void **) &table_D_gpu, TABLE_SIZE * sizeof(u16));
    cudaMalloc((void **) &table_E_gpu, memory_size * sizeof(u16));
    // copy tables BC, D and calculate table E
    cudaMemcpy(table_BC_gpu, table_BC_cpu, TABLE_SIZE * sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(table_D_gpu, table_D_cpu, TABLE_SIZE * sizeof(u16), cudaMemcpyHostToDevice);
    simple_gpu<<<memory_size / block_size, block_size>>>(table_E_gpu);
    // allocate gpu memory for result
    u16 *gpu_res_avg;
    u16 *gpu_res_min;
    u16 *gpu_res_max;
    cudaMalloc(&gpu_res_avg, n_batches_per_job * sizeof(u16));
    cudaMalloc(&gpu_res_min, n_batches_per_job * sizeof(u16));
    cudaMalloc(&gpu_res_max, n_batches_per_job * sizeof(u16));
    cudaDeviceSynchronize();
    // check for memory errors
    auto code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
        return 1;
    } else {
        end = getSecond();
        std::cout << "Finished init in " << end - start << " seconds" << std::endl;
    }

    start = getSecond();
    for (int j = 0; j < n_jobs; j++) {
        u64 job_offset = offset + j * job_batch_size;
        gpu_LUT<<<grid_size, block_size>>>(gpu_res_avg, gpu_res_min, gpu_res_max, table_BC_gpu, table_D_gpu,
                                           table_E_gpu, memory_size, job_offset, n_batches_per_job);
        cudaDeviceSynchronize();

        // COPY BACK
        u64 copy_back_offset = j * n_batches_per_job;
        cudaMemcpy(cpu_res_avg + copy_back_offset, gpu_res_avg, n_batches_per_job * sizeof(u16),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_res_min + copy_back_offset, gpu_res_min, n_batches_per_job * sizeof(u16),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_res_max + copy_back_offset, gpu_res_max, n_batches_per_job * sizeof(u16),
                   cudaMemcpyDeviceToHost);

        // CHECK CUDA ERRORS
        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cout << "Error: " << cudaGetErrorString(code) << std::endl;
            return 1;
        }

        std::cout << "Finished GPU job " << j + 1 << " of " << n_jobs << std::endl;
    }

    end = getSecond();
    std::cout << "Time: " << end - start << std::endl;

    // save the three results each to a bin file
    std::ofstream out_avg("output/avg.bin", std::ios::binary);
    std::ofstream out_min("output/min.bin", std::ios::binary);
    std::ofstream out_max("output/max.bin", std::ios::binary);
    out_avg.write((char *) cpu_res_avg, tot_batches * sizeof(u16));
    out_min.write((char *) cpu_res_min, tot_batches * sizeof(u16));
    out_max.write((char *) cpu_res_max, tot_batches * sizeof(u16));

    // print first three values of each result
    std::cout << "First three values of avg: " << cpu_res_avg[0] << " " << cpu_res_avg[1] << " " << cpu_res_avg[2]
              << std::endl;
    std::cout << "First three values of min: " << cpu_res_min[0] << " " << cpu_res_min[1] << " " << cpu_res_min[2]
              << std::endl;
    std::cout << "First three values of max: " << cpu_res_max[0] << " " << cpu_res_max[1] << " " << cpu_res_max[2]
              << std::endl;

    // FREE memory
    cudaFree(gpu_res_avg);
    cudaFree(gpu_res_min);
    cudaFree(gpu_res_max);
    cudaFree(table_BC_gpu);
    cudaFree(table_D_gpu);
    cudaFree(table_E_gpu);
    free(table_BC_cpu);
    free(table_D_cpu);
    free(cpu_res_avg);
    free(cpu_res_min);
    free(cpu_res_max);

    return 0;
}