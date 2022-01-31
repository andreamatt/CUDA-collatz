#ifndef STATS_CUH
#define STATS_CUH

double average(double *data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

double max(double *data, int n) {
    double max = data[0];
    for (int i = 0; i < n; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

double min(double *data, int n) {
    double min = data[0];
    for (int i = 0; i < n; i++) {
        if (data[i] < min) {
            min = data[i];
        }
    }
    return min;
}

int print_stats(double *cpu, double *cpu_al, double *gpu, double *gpu_al, double *gpu_cp, int n_tests) {
    double cpu_avg = average(cpu, n_tests);
    double cpu_max = max(cpu, n_tests);
    double cpu_min = min(cpu, n_tests);
    double cpu_al_avg = average(cpu_al, n_tests);
    double cpu_al_max = max(cpu_al, n_tests);
    double cpu_al_min = min(cpu_al, n_tests);
    double gpu_avg = average(gpu, n_tests);
    double gpu_max = max(gpu, n_tests);
    double gpu_min = min(gpu, n_tests);
    double gpu_al_avg = average(gpu_al, n_tests);
    double gpu_al_max = max(gpu_al, n_tests);
    double gpu_al_min = min(gpu_al, n_tests);
    double gpu_cp_avg = average(gpu_cp, n_tests);
    double gpu_cp_max = max(gpu_cp, n_tests);
    double gpu_cp_min = min(gpu_cp, n_tests);
    printf("CPU: %.3f %.3f %.3f\n", cpu_avg, cpu_min, cpu_max);
    printf("CPU_AL: %.3f %.3f %.3f\n", cpu_al_avg, cpu_al_min, cpu_al_max);
    printf("GPU: %.3f %.3f %.3f\n", gpu_avg, gpu_min, gpu_max);
    printf("GPU_AL: %.3f %.3f %.3f\n", gpu_al_avg, gpu_al_min, gpu_al_max);
    printf("GPU_CP: %.3f %.3f %.3f\n", gpu_cp_avg, gpu_cp_min, gpu_cp_max);
    printf("GPU_total: %.3f %.3f %.3f\n", gpu_avg + gpu_al_avg + gpu_cp_avg, gpu_min + gpu_al_min + gpu_cp_min,
           gpu_max + gpu_al_max + gpu_cp_max);
    return 0;
}

#endif