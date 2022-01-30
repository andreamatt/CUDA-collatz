#include <iostream>
#include "types.cu"

__global__ void sum_array(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] + idx % 5;
    }
}

void warmup() {
    double start = getSecond();
    int n = 1 << 27;

    int *h_A = (int *) malloc(n * sizeof(int));
    int *h_B = (int *) malloc(n * sizeof(int));

    auto initCPUData = [](int *ip, int size) {
        time_t t;
        srand((unsigned) time(&t));
        for (int i = 0; i < size; i++) {
            ip[i] = rand() % 2000;
        }
    };

    initCPUData(h_A, n);
    initCPUData(h_B, n);


    cudaFree(nullptr);

    int *a, *b, *c;
    cudaMalloc(&a, n * sizeof(int));
    cudaMalloc(&b, n * sizeof(int));
    cudaMalloc(&c, n * sizeof(int));

    cudaMemcpy(a, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(c, 0, n * sizeof(int));

    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;
//    std::cout << "block_size: " << block_size << std::endl;
//    std::cout << "grid_size: " << grid_size << std::endl;
    sum_array<<<grid_size, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(h_A);
    free(h_B);

    double end = getSecond();
//    std::cout << "Warmup time: " << end - start << std::endl;
}