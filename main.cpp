// implement same code as main.cu, but for parallel CPU execution
#include <iostream>
#include <fstream>
#include "utils/time.cu"
#include "cpu.cu"
#include <vector>
#include <algorithm>
#include <execution>

#define BITS 12
#define MEMORY_POWER 31
#define BATCH_SIZE 1024

void generate_LUT(u64 *BC_table, u16 *D_table, u64 table_size) {
    u32 A = table_size;
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

void generate_memory(u16 *table_E, u64 n, u64 *table_BC, u16 *table_D, u64 table_size) {
    table_E[0] = 0;
    for (u64 i = 1; i < n; i++) {
        u64 num = i;
        u16 count = 0;
        // use table if possible (num must be bigger than table_size)
        while (num >= table_size) {
            // if already calculated
            if (num < i) {
                count += table_E[num];
                num = 0;
                break;
            }

            u64 n_high = num >> BITS;
            u64 n_low = num - (n_high << BITS);
            u64 bc = table_BC[n_low];
            u32 b = (bc >> 16) >> 16;
            u32 c = bc - ((((u64) b) << 16) << 16);
            count += table_D[n_low];
            num = n_high * b + c;
        }

        // then loop without table
        while (num > 1) {
            if (num % 2 == 0) {
                num = num / 2;
                count++;
            } else {
                num = (3 * num + 1) / 2;
                count += 2;
            }
        }

        table_E[i] = count;
    }
}

int main() {
    double start, end;
    start = getSecond();

    u64 N_to_calc = power(34);
    u64 offset = power(40);
    u64 tot_batches = N_to_calc / BATCH_SIZE;
    u64 table_size = power(BITS);
    u64 n_jobs = 16;
    u64 job_batch_size = N_to_calc / n_jobs;
    u64 batches_per_job = tot_batches / n_jobs;

    u64 memory_size = power(MEMORY_POWER);

    // calculate LUTs for cpu
    std::cout << "Init started" << std::endl;
    u64 *table_BC = (u64 *) malloc(table_size * sizeof(u64));
    u16 *table_D = (u16 *) malloc(table_size * sizeof(u16));
    generate_LUT(table_BC, table_D, table_size);
    std::cout << "LUT generation finished" << std::endl;
    // calculate table_E for cpu
    u16 *table_E = (u16 *) malloc(memory_size * sizeof(u16));
    generate_memory(table_E, memory_size, table_BC, table_D, table_size);
    std::cout << "Memory generation finished" << std::endl;

    u16 *cpu_res_avg = (u16 *) malloc(tot_batches * sizeof(u16));
    u16 *cpu_res_min = (u16 *) malloc(tot_batches * sizeof(u16));
    u16 *cpu_res_max = (u16 *) malloc(tot_batches * sizeof(u16));

    end = getSecond();
    std::cout << "Init finished in " << end - start << " seconds" << std::endl;

    start = getSecond();
    // parallel loop
    auto jobs = std::vector<u64>(n_jobs);
    for (u64 i = 0; i < n_jobs; i++) {
        jobs[i] = i;
    }

    std::for_each(
            std::execution::par_unseq,
            jobs.begin(),
            jobs.end(),
            [
                    &job_batch_size,
                    &memory_size,
                    &batches_per_job,
                    &offset,
                    &cpu_res_avg,
                    &cpu_res_min,
                    &cpu_res_max,
                    &table_BC,
                    &table_D,
                    &table_E,
                    &table_size
            ](u64 job_i) {
                std::cout << "Job " << job_i << " started" << std::endl;
                u64 start_batch = job_i * job_batch_size;
                u64 end_batch = (job_i + 1) * job_batch_size;
                for (u64 batch_idx = 0; batch_idx < batches_per_job; batch_idx++) {
                    u64 start_i = start_batch + batch_idx * BATCH_SIZE;
                    u64 end_i = start_i + BATCH_SIZE;
                    u16 min_c = UINT16_MAX;
                    u16 max_c = 0;
                    u32 sum_c = 0;

                    for (u64 i = start_i; i < end_i; i++) {
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

                        count += table_E[num];

                        if (count < min_c) {
                            min_c = count;
                        }
                        if (count > max_c) {
                            max_c = count;
                        }
                        sum_c += count;
                    }

                    cpu_res_avg[job_i * batches_per_job + batch_idx] = sum_c / BATCH_SIZE;
                    cpu_res_min[job_i * batches_per_job + batch_idx] = min_c;
                    cpu_res_max[job_i * batches_per_job + batch_idx] = max_c;
                }
            }
    );


    end = getSecond();
    std::cout << "Time: " << end - start << std::endl;

    // save the three results each to a bin file
    std::ofstream out_avg("output/avg_cpu.bin", std::ios::binary);
    std::ofstream out_min("output/min_cpu.bin", std::ios::binary);
    std::ofstream out_max("output/max_cpu.bin", std::ios::binary);
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
    free(table_BC);
    free(table_D);
    free(table_E);
    free(cpu_res_avg);
    free(cpu_res_min);
    free(cpu_res_max);

    return 0;
}