#include "utils/types.cu"


__global__ void simple_gpu(u16 *res, u64 n){
    u64 i = blockIdx.x * blockDim.x + threadIdx.x +1;
    if(i < n){
        u64 a = i;
        u16 c = 0;
        while (a != 1) {
            if (a % 2 == 0) {
                a = a / 2;
                c++;
            } else {
                a = (3 * a + 1) / 2;
                c+=2;
            }
        }
        res[i] = c;
    }
}

//void dynamic_cpu(u16 *res, u64 n){
//    res[0] = 0;
//    for (u64 i = 1; i < n; i++) {
//        u64 a = i;
//        u16 c = 0;
//        while (a != 1) {
//            if (a < i){
//                c += res[a];
//                break;
//            }
//
//            if (a % 2 == 0) {
//                a = a / 2;
//                c++;
//            } else {
//                a = (3 * a + 1) / 2;
//                c+=2;
//            }
//        }
//        res[i] = c;
//    }
//}