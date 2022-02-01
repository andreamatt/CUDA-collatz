#include "utils/types.cu"


__global__ void simple_gpu(u16 *res, u64 n) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0) {
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
    } else if (i == 0) {
        res[i] = 0;
    }
}

__global__ void simple_gpu_batchsave(u16 *res, u64 n, u64 n_batches, u32 batch_size) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0) {
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
    } else if (i == 0) {
        res[i] = 0;
    }
}

__global__ void simple_gpu32(u32 *res, u64 n) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i > 0) {
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
    } else if (i == 0) {
        res[i] = 0;
    }
}

__global__ void simple_gpu_offset(u16 *res, u64 n, u64 offset) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    u64 a = i + offset;
    if (a < n && a > 0) {
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
    }
}

__global__ void simple_gpu_multiple(u16 *res, u64 n, u32 m) {
    u64 i_start = m * (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    for (u32 j = 0; j < m; j++) {
        u64 i = i_start + j;
        if (i < n) {
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
        }
    }
}

__global__ void simple_gpu_batch_1(u16 *res, u64 n, u64 n_batches, u32 batch_size) {
    u64 id = blockIdx.x * blockDim.x + threadIdx.x;
    u64 i_start = batch_size * id;
    u64 i_end = i_start + batch_size;
    u16 min_c = UINT16_MAX;
    u16 max_c = 0;
    u32 sum_c = 0;
    for (u64 i = i_start; i < i_end; i++) {
        if (i < n && i > 0) {
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
            if (c < min_c) {
                min_c = c;
            }
            if (c > max_c) {
                max_c = c;
            }
            sum_c += c;
        }
    }
    res[id] = sum_c / batch_size;
    res[id + n_batches] = min_c;
    res[id + n_batches * 2] = max_c;
}

__global__ void simple_gpu_batch_2(u16 *res, u64 n, u64 n_batches, u32 batch_size) {
    u64 id = blockIdx.x * blockDim.x + threadIdx.x;
    u64 i_start = batch_size * id;
    u64 i_end = i_start + batch_size;
    u16 min_c = UINT16_MAX;
    u16 max_c = 0;
    u32 sum_c = 0;
    for (u64 i = i_start; i < i_end; i++) {
        if (i < n && i > 0) {
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
            if (c < min_c) {
                min_c = c;
            }
            if (c > max_c) {
                max_c = c;
            }
            sum_c += c;
        }
    }
    res[id] = sum_c / batch_size;
    res[id + n_batches] = min_c;
    res[id + n_batches * 2] = max_c;
}