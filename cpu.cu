#include <iostream>
#include "utils/types.cu"

void compare_arrays(u16 *a, u16 *b, u64 n) {
    for (u64 i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            std::cout << "Error: a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << std::endl;
            return;
        }
    }
}

void verify_cpu(u64 n) {
    for (u64 i = 1; i < n; i++) {
        u64 a = i;
        while (a != 1) {
            if (a < i) {
                break;
            }

            if (a % 2 == 0) {
                a = a / 2;
            } else {
                a = (3 * a + 1) / 2;
            }
        }
    }
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

void simple_cpu_scaled(u16 *res_avg, u16 *res_min, u16 *res_max, u64 n, u32 scale) {
    u64 n_scaled = n / scale;
    for (int s = 0; s < scale; s++) {
        u64 i_start = s * n_scaled;
        u64 i_end = (s + 1) * n_scaled;
        u16 min_c = U16_MAX;
        u16 max_c = 0;
        u32 sum_c = 0;
        for (u64 i = i_start; i < i_end; i++) {
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
        res_avg[s] = sum_c / n_scaled;
        res_min[s] = min_c;
        res_max[s] = max_c;
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