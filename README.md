# Collatz in CUDA

- [Collatz in CUDA](#collatz-in-cuda)
  - [1. Introduction](#1-introduction)
  - [2. Specific problem](#2-specific-problem)
  - [3. Hardware specifications](#3-hardware-specifications)
  - [4. Multiple elements per thread](#4-multiple-elements-per-thread)
  - [5. Batch reduction](#5-batch-reduction)
  - [6. Coalesced access](#6-coalesced-access)
  - [7. LUTs and dynamic programming](#7-luts-and-dynamic-programming)
  - [8. Memory usage](#8-memory-usage)
    - [Option 0: No LUTs](#option-0-no-luts)
    - [Option 1: Only table E in shared memory](#option-1-only-table-e-in-shared-memory)
    - [Option 2: All in shared memory](#option-2-all-in-shared-memory)
    - [Option 3: Unify B and C](#option-3-unify-b-and-c)
    - [Option 4: Only table E, in global memory](#option-4-only-table-e-in-global-memory)
    - [Option 5: BCD in shared, E in global](#option-5-bcd-in-shared-e-in-global)
  - [9. Final version](#9-final-version)
  - [10. CPU version](#10-cpu-version)
  - [11. Visualizations](#11-visualizations)
  - [12. Conclusion](#12-conclusion)

## 1. Introduction
The Collatz conjecture is an open mathematical problem based on simple operations on natural numbers.
It is based on the two following rules:
- if n is even, divide it by 2
- if n is odd, multiply by 3 and add 1

For example, the starting number 7 generates the sequence *22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 4, 2, 1, ...*

The conjecture is: given any starting positive integer, applying the two rules recursively will eventually lead to 1. Once 1 is reached, a cycle is formed ***1, 4, 2, 1***.
As of now, the conjecture is believed to be true but there is no concrete proof. All numbers up to 2^68 have been checked and no counter-example was found. There are two types of possible counter-example:
- a sequence that ends in a cycle different than *1, 4, 2, 1*
- a sequence that increases without bound

An associated problem is to look at the number of steps that it takes to converge to 1. For examples, 7 takes 16 steps, while the next natural number, 8, takes only 3.
Plotting the number of steps leads to interesting patterns:
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Collatz-stopping-time.svg/1024px-Collatz-stopping-time.svg.png" style="zoom:50%;" />
([source](https://en.wikipedia.org/wiki/Collatz_conjecture))

Let's define the process of applying a single step as $f(n)$:
$$
f(n)={\begin{cases}{
 \frac{n}{2}}&\text{if }n\equiv 0\pmod{2}\\
 3n+1&\text{otherwise}
 \end{cases}}
$$

and the number of steps to reach 1 as $steps(n)$:
$$
 steps(n)={\begin{cases}
 0&{\text{if }}n = 1\\
 1+steps(f(n))&\text{otherwise}
 \end{cases}}
$$

If n is odd, the resulting 3n+1 is even, so the odd step can be redefined as an odd->even step. From now on, this optimization is always applied in the code, but is ignored in the explanations for clarity.

The function $steps(n)$ has two interesting properties:
- steps(x) and steps(y) can be calculated independently, therefore **enabling parallel computation**
- given its recursive nature, **dynamic programming can be applied** when calculating multiple values

These observations make the $steps$ function a good yet challenging candidate for GPU computation. The parallelization should allow a lot more throughput, but given the different sequences that each number produces, the dynamic programming part is not obvious. Since two adjacent numbers can go two completely different and far points, the memory access pattern is crucial.

## 2. Specific problem
The initial problem was calculating $step(n)$ for each natural number up to a certain limit. This quickly proved to be infeasible simply due to the high amount of data. Even using 16-bit unsigned integers for memorizing the results, a basic gpu implementation can calculate 10^9 numbers per second, generating about 2 GB/s of throughput. To avoid high data amounts, the problem is reformulated as follows:
given a batch size *BATCH_SIZE*, a starting offset *OFFSET* and an amount of numbers to compute *N*,
calculate the average, minimum and maximum value of $step(n)$ in each batch of consecutive numbers from OFFSET to OFFSET+N.
A batch size of 1024 is enough to reduce the throughput significantly, while still allowing general behavior to emerge (as will be shown in the plots of section [11](#11.-Visualizations)).
Since the number of steps doesn't reach 2^16 for any of the first 2^64 numbers ([source](https://en.wikipedia.org/wiki/Collatz_conjecture#Empirical_data)), 16-bit unsigned integers can be used safely. From now on, **u16**, **u32** and **u64** are 16, 32 and 64-bit unsigned integers.
Each batch produces 3 values (average, min, max), therefore the total amount of data generated is $16\ bit * \frac{N}{BATCH\_SIZE} *3$

## 3. Hardware specifications
All the algorithms are implemented in CUDA 11.6 and tested on an NVIDIA GTX 1660 SUPER, with the following relevant specs:
- CUDA Capability 7.5
- Total global memory: 6144 MB
- 22 Multiprocessors, with 64 CUDA cores each
- Total of 1408 CUDA cores running at a maximum of 1785 MHz
- Memory clock: 7001 MHz
- Memory BUS width: 192 bit
- Maximum shared memory per block: 49152 bytes
- Warp size: 32
- Maximum threads per block: 1024
- Supported shared memory bank configurations: 4 byte only (does not support 8 byte)


## 4. Multiple elements per thread
The first CUDA implementation assigned a thread to each number. Since the length of the sequence differs for each thread, some divergence is expected when some threads of the 32-thread warp have finished, while others haven't. To verify and mitigate the effect, I tried assigning multiple values to the same thread. If the total sequence length is similar between threads, they should take similar time to execute.
The unrolling technique was tested with different numbers of iterations per thread, all with similar results.
Calculating 2^30 values and averaging over 10 runs, the basic approach takes 1.21 seconds.
After tuning the amount of numbers for each thread to 32 (the best result), the modified approach takes 1.29.
This implies that either divergence has very little impact on the performance or unrolling doesn't help.
Since the difference between unrolling a few times and unrolling completely is negligible, the unrolled version is tested in the next section.

> The corresponding code is test1.1.cu and test1.2.cu


## 5. Batch reduction
Given the transition to batches described in [section 2](#2.-Specific-problem), there are two main possibilities: either each thread calculates an entire batch, or each thread calculates a single item and a reduction is later required. Partial unrolling was not considered given the small difference in performance between complete and partial unrolling raised in the previous section.
The reduction for the average requires summing all elements in a batch, then dividing by BATCH_SIZE. The summation overflows with u16, so u32 are required.
Calculating 2^29 elements, one per thread, average over 50 runs:
- compute using u16 (only for reference): 0.60 seconds
- compute using u32: 0.63 seconds
- compute using u16, then copy to u32: 0.65 seconds

> Code at test2.cu

This implies that when reduction is necessary, it is better to do all the computations using u32, rather than copying to u16 later.

The next comparison is therefore between:
- 3.1: 1 batch per thread, using u16
- 3.2: 1 item per thread using u32, then reduce to batches
- 3.3: same as 3.2, but alternative reduction

Calculating 2^30 numbers, average over 5 runs, time in seconds:
| batch size | 3.1      | 3.2  | 3.3  |
| ---------- | -------- | ---- | ---- |
| 256        | 1.26     | 1.85 |      |
| 512        | 1.25     | 2.36 |      |
| **1024**   | **1.25** | 2.35 | 2.22 |
| 2048       | 1.29     | 2.20 |      |
| 4096       | 1.31     | 2.23 |      |
| 8192       | 1.32     | 2.25 |      |
| 16384      | 1.33     | 2.59 |      |
| 32768      | 1.74     | 2.37 |      |
| 65536      | 1.77     | 2.00 |      |
| 131072     | 3.49     | 1.67 |      |
| 262144     | 6.95     | 1.95 |      |
| 524288     | 13.7     | 2.47 |      |
| 1048576    | 26.9     | 3.64 |      |

3.2 is compared against 3.1 at all batch sizes only for reference; the key result is at batch size 1024.

To choose the best reduction technique, I compared all the optimizations described on the [CUDA docs about reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), thanks to the implementation code available [here](https://github.com/deeperlearning/professional-cuda-c-programming).
Such techniques do not scale very well with the many but small reductions required here, so I also tested a simpler technique: in 3.2 a thread sums all elements of a batch. Instead, in 3.3 i used reduceGmemUnroll, which does a 4-times unroll, followed by complete unroll for-loop, using global memory. There was a shared memory version as well, but the shared memory cannot be used, as it is reserved for the lookup tables introduced in the next section.

> Code at test3.1.cu, test3.2.cu, test3.3.cu, reduce_test.cu and test_reduction.cu

**At the chosen batch size of 1024, the best approach is the unrolled version, where each thread calculates an entire batch.**

## 6. Coalesced access
Given the number of batches N_batches, the final result can be organized in two different ways:
- a single array [avg_0, min_0, max_0, avg_1, min_1, max_1, ...] with 3\*N_batches values
- three arrays [avg_0, avg_1, ...], [min_0, min_1, ...], [max_0, max_1, ...] with N_batches values each

When all the threads in a warp finish calculating the avg,min,max of their respective batch, writing back the result to global memory can be coalesced by using the three array variant. With that, consecutive threads access consecutive memory cells.

## 7. LUTs and dynamic programming
As explained in [this paper](https://www.researchgate.net/publication/300359600_GPU-Accelerated_Verification_of_the_Collatz_Conjecture), it is possible to accelerate the procedure using lookup-tables (LUTs) that group multiple iterations into one.
This is based on the observation that the last *d* bits of an item determine some of the next iterations. Let's consider *n* as the current value to be calculated. If *d=2*, *n=4h+l*, where *l* are the least significant 2 bits and *h* are the remaining bits.
The resulting new value of n can be calculated based on the following values of *l*:
- 00: *n=4h+0* => apply two even operations => *4h+0 => 2h => h* 
- 01: *n=4h+1* => apply one odd and two even operations => *4h+1 => 12h+4 => 6h+2 => 3h+1*
- 10: *n=4h+2* => apply even, then odd, then even => *4h+2 => 2h+1 => 6h+4 => 3h+2*
- 11: *n=4h+3* => apply odd, even, odd, even => *4h+3 => 12h+10 => 6h+5 => 18h+16 => 9h+8*

When applying even/odd rules to *h* and *l* we have to stop if the multiplication factor of *h* is odd, as it is impossible to decide which rule to apply without knowing the value of *h*.
This results in the following LUT:
|     | B   | C   |
| --- | --- | --- |
| 00  | 1   | 0   |
| 01  | 3   | 1   |
| 10  | 3   | 2   |
| 11  | 9   | 8   |

So, given a number *n = 4h+l*, the next number is $n = B[l]*h + C[l]$.

This can be generalized to any *d* bits, with the caveat that the size of the table is proportional to the number of possible values of *l*: $Size(Table) = k*2^d$, where *k* is the size of one entry.
The general case defines $n = 2^d*h + l$. The coefficients *b* and *c* determine the steps taken and the final table entry, given the value of *l*. The procedure for generating the final table entry is to apply the following rules until *b* is odd:
- even rule: if both *b* and *c* are even, divide them by two
- odd rule: if *b* is even and *c* is odd, triple *b*, triple *c* and add 1 to *c*

Tables B and C speed up the verification of the conjecture, but an extra table D is required to calculate the number of steps. Table D indicates how many steps were skipped using the optimization. The modified version of the previous examples generates this:
|     | B   | C   | D   |
| --- | --- | --- | --- |
| 00  | 1   | 0   | 2   |
| 01  | 3   | 1   | 3   |
| 10  | 3   | 2   | 3   |
| 11  | 9   | 8   | 4   |

An important note is that the optimization makes sense only for $n \geq 2^d$. For instance, using *d=2* and *n=2*: *h=0* and *l=2* => $n=0*3 + 2 = 2$. This loop is easily avoidable by having a fourth table E, which enumerates the number of steps required to reach 1, given that *n=l*.
So the modified version is now:
|     | B   | C   | D   | E   |
| --- | --- | --- | --- | --- |
| 00  | 1   | 0   | 2   | 0   |
| 01  | 3   | 1   | 3   | 0   |
| 10  | 3   | 2   | 3   | 1   |
| 11  | 9   | 8   | 4   | 7   |

Since the value of $steps(n)$ is undefined at *n=0*, we set $E[0] = 0$ for simplicity but the value should never be read.

There is a distinction to be made between tables B/C/D and table E: the first three need to be read at each iteration, while table E is read only when $n < 2^d$. This means that while B,C and D benefit a lot from being on fast memory, table E doesn't. Instead, E could be increased in size to include more values. Let's define $M=2^m$ as the size of the table E. Once $n < M$, the procedure reads the value of $E[n]$ and stops. Therefore, *m* must be at least *d*, but it can be as large as the chosen memory allows.

## 8. Memory usage
The final comparison revolves around the usage, sizes and location of the four tables B,C,D,E.
Each entry of tables B and C requires 32-bit unsigned integers (the coefficients *b* and *c* can get large), while D and E need only 16-bit ones (as stated before, the number of steps does not reach 2^16).
Reminder: the available shared memory is 49152 bytes, while the global memory is 6 GB.

The chosen benchmark has the following parameters:
- N = 2^34
- OFFSET = 2^40
- BATCH_SIZE = 1024

The resulting time is always in seconds and does not include the table generation time.

> Code for these tests is in test4.0.cu, test4.1.cu, ...

### Option 0: No LUTs
As a reference, the algorithm without any LUT optimization takes **24.2** seconds.

### Option 1: Only table E in shared memory
No BCD tables mean no optimization in the *steps(n)* procedure, but it stops sooner thanks to table E.
Each table entry takes up 2 bytes.

| m      | M / table size | Shared mem (bytes) | Time     |
| ------ | -------------- | ------------------ | -------- |
| 10     | 1024           | 2048               | 19.6     |
| 11     | 2048           | 4096               | 19.4     |
| 12     | 4096           | 8192               | 19.0     |
| 13     | 8192           | 16384              | 18.5     |
| **14** | **16384**      | 32768              | **17.9** |

### Option 2: All in shared memory
Tables B, C, D and E all have the same size and are stored in shared memory.
Each table entry takes up 4 (B) + 4 \(C\) + 2 (D) + 2 (E) = 12 bytes.
| d=m    | Table size | Shared mem (bytes) | Time     |
| ------ | ---------- | ------------------ | -------- |
| 2      | 4          | 48                 | 12.4     |
| 3      | 8          | 96                 | 8.24     |
| 4      | 16         | 192                | 6.14     |
| 5      | 32         | 384                | 4.83     |
| 6      | 64         | 768                | 4.73     |
| 7      | 128        | 1536               | 4.74     |
| 8      | 256        | 3072               | 4.38     |
| 9      | 512        | 6144               | 3.85     |
| 10     | 1024       | 12288              | 3.44     |
| 11     | 2048       | 24576              | 3.21     |
| **12** | **4096**   | 49152              | **3.11** |


### Option 3: Unify B and C
As of now, each iteration would read 3 different values: $B[l]$, $C[l]$ and $D[l]$ using three different instructions. Instead, tables B and C can be merged into table BC, where each value takes 64 bits and is $BC[l] = 2^{32} * B[l] + C[l]$. Access to the tables is not coalesced (the values of *l* are chaotic), but this optimization saves some round-trips, as observed below:
| d=m    | Table size | Shared mem (bytes) | Time     |
| ------ | ---------- | ------------------ | -------- |
| 2      | 4          | 48                 | 10.6     |
| 3      | 8          | 96                 | 7.12     |
| 4      | 16         | 192                | 5.33     |
| 5      | 32         | 384                | 5.14     |
| 6      | 64         | 768                | 4.71     |
| 7      | 128        | 1536               | 4.63     |
| 8      | 256        | 3072               | 4.01     |
| 9      | 512        | 6144               | 3.47     |
| 10     | 1024       | 12288              | 3.07     |
| 11     | 2048       | 24576              | 2.87     |
| **12** | **4096**   | 49152              | **2.82** |

For all possible table sizes, this version is faster than the previous.
A possible improvement would be to use eight bytes per bank, given that each entry in BC takes 8 bytes. I could not test this, because the GPU used does not support that mode.


### Option 4: Only table E, in global memory
Here, tables B,C and D are missing, but table E is very big, allowing to stop the iterations sooner.
| m   | Table size | Global memory | Time     |
| --- | ---------- | ------------- | -------- |
| 10  | 2^10       | 2KB           | 20.1     |
| 12  | 2^12       | 8KB           | 19.2     |
| 15  | 2^15       | 64KB          | 18.0     |
| 20  | 2^20       | 2MB           | 15.7     |
| 25  | 2^25       | 64MB          | 13.2     |
| 30  | 2^30       | 2GB           | 10.0     |
| 31  | **2^31**   | 4GB           | **9.34** |

This version is better compared to the no-LUTs one ([option 0](#Option-0-No-LUTs)) and the shared memory one ([option 1](#Option-1-Only-table-E-in-shared-memory), but significantly worse compared to the previous two.

### Option 5: BCD in shared, E in global
The next step is to use both memories:
- tables BC and D in shared memory as they are accessed frequently (at each iteration step)
- table E very big and in global memory

Tables B and C are merged as described in [option 3](#Option-3-Unify-B-and-C) for higher performance.
The best parameters for each previous version are simply the highest allowed by the memory:
- ***d = 12*** => 4096 entries for BC and D, total 40960 bytes
- ***m = 31*** => 2^31 entries for E, total of 4 GB

The resulting time is **1.72** seconds.

I did not test using all the 6 GB available for table E because some of the VRAM (0.5 to 1 GB) is used by the Windows operating system. A future improvement could be to test the same hardware on a headless linux server.

## 9. Final version
The best performing version is therefore using look-up tables to skip iterations and a dynamic-programming approach to stop iterations way before reaching 1.
In order to calculate larger amounts of numbers, for instance 2^40, the best approach is to divide the work in batches. A good parameter for the job batch size is 2^36, as a single job takes about 10 seconds, then the device is synchronized with the host and the data is copied to the host memory.

Here are some results with the final version:
| N    | OFFSET | Time (seconds) |
| ---- | ------ | -------------- |
| 2^36 | 0      | 21.57          |
| 2^37 | 0      | 34.4           |
| 2^38 | 0      | 50.8           |
| 2^39 | 0      | 78.2           |
| 2^40 | 0      | 134            |
| 2^36 | 2^20   | 21.2           |
| 2^37 | 2^20   | 34.3           |
| 2^38 | 2^20   | 50.6           |
| 2^39 | 2^20   | 77.4           |
| 2^40 | 2^20   | 136            |
| 2^36 | 2^40   | 7.35           |
| 2^37 | 2^40   | 14.2           |
| 2^38 | 2^40   | 28.7           |
| 2^39 | 2^40   | 57.8           |
| 2^40 | 2^40   | 116            |

> Code on main.cu


## 10. CPU version
I implemented the final procedure for the CPU too, using C++.
I introduced parallelism using the parallel for-loop introduced with C++ 17 ([described here](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)).
All the LUTs are on RAM and have the same size as in the GPU version.
The CPU used is an AMD Ryzen 3 3100 (4 cores, 8 threads), clocked at 3.9 GHz.

Using the usual benchmark of 2^34 numbers with 2^40 offset, the resulting time (seconds) is:
|                    | CPU      | GPU      |
| ------------------ | -------- | -------- |
| Table init time    | 23.5     | 2.56     |
| **Execution time** | **57.9** | **1.74** |

This means that the GPU version is about 30 times faster. Of course, the CPU version could be tweaked to improve its performance, but that is outside the scope of this project.

> Code on main.cpp

## 11. Visualizations
The generated data is saved on disk in binary files. Those can be read using python and plotted using [numpy](https://numpy.org/) and [seaborn](https://seaborn.pydata.org/). Given the amount of points generated, not all of them can be included, as the libraries cannot handle such large quantities.

The following visualizations use N=2^36 and offset=0.

Plotting only the first 100000 points:
![](https://i.imgur.com/cW8gAnt.jpg)
![](https://i.imgur.com/uL0BJ4K.jpg)
![](https://i.imgur.com/kdXLxa8.jpg)
![](https://i.imgur.com/xdFS9yz.jpg)

Plotting a point every 1000 (about 67k points):
![](https://i.imgur.com/bHPOcBa.jpg)
![](https://i.imgur.com/0wg1pIp.jpg)
![](https://i.imgur.com/8XMfgYj.jpg)
![](https://i.imgur.com/aANAS1W.jpg)

There are some visible patterns both in the max and min metrics, very similar to the pattern of the normal Collatz step calculation visible in the [introduction](#1-Introduction).

> Code at visualize.py

## 12. Conclusion
Despite the complexities deriving from the unpredictability of the Collatz sequences, the parallelization and the dynamic programming can accelerate the procedure significantly.
Further improvements could be applied with small instruction optimizations or by introducing a CPU-GPU cooperative approach.
Given the relatively small BUS width of 192 bits of my GPU, higher-end hardware could improve performance significantly given the importance of memory throughput in this application.
At the same time, the size of the BC and D LUTs cannot be increased, as shared memory size is the same across existing models.
An interesting extension would be to generate the plots too using the GPU, instead of saving the data and creating them afterwards.
In general, the dynamic programming approach seems applicable to GPU problems, especially if it can take advantage of small lookup tables that fit into shared memory.

