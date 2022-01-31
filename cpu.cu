#include <iostream>
#include "utils/types.cu"

bool compare_arrays(u16 *a, u16 *b, u64 n) {
    for (u64 i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            std::cout << "Error: a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void simple_cpu(u16 *res, u64 n) {
    res[0] = 0;
    for (u64 i = 1; i < n; i++) {
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

void simple_cpu_batch(u16 *res, u64 N, u64 n_batches, u32 batch_size) {
    for (int b = 0; b < n_batches; b++) {
        u64 i_start = b * batch_size;
        u64 i_end = (b + 1) * batch_size;
        u16 min_c = UINT16_MAX;
        u16 max_c = 0;
        u32 sum_c = 0;
        for (u64 i = i_start; i < i_end; i++) {
            if (i < N && i > 0) {
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

        res[b] = sum_c / batch_size;
        res[b + n_batches] = min_c;
        res[b + n_batches * 2] = max_c;
    }
}

void dynamic_cpu(u16 *res, u64 n) {
    res[0] = 0;
    for (u64 i = 1; i < n; i++) {
        u64 a = i;
        u16 c = 0;
        while (a != 1) {
            if (a < i) {
                c += res[a];
                break;
            }

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