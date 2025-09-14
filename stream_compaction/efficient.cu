#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int N, int offset, int* iData) {
            int offDown1 = offset >> 1;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            index *= offset;
            index += offDown1 - 1;
            if (index + offDown1 >= N) {
                return;
            }
            if (index + offDown1 == 6) {
                printf("Huh?:Old %d, New %d ", iData[index + offDown1], iData[index] + iData[index + offDown1]);
            }

            iData[index + offDown1] = iData[index] + iData[index + offDown1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int* dev_idata;
            int pow2N = ilog2ceil(n);
            int size = 1 << pow2N;
            int* padded = new int[size];
            for (int i = 0; i < size; i++) {
                if (i >= n || i == 0) {
                    padded[i] = 0;
                }
                else {
                    padded[i] = idata[i];
                }
            }

            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            cudaMemcpy(dev_idata, padded, size * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 64;
            int gridSize = (size + blockSize - 1) / blockSize;
            int offset = 2;
            for (int d = 0; d < pow2N; d++) {
                offset = 1 << d + 1;
                upSweep << <gridSize, blockSize >> > (size, offset, dev_idata);
            }

            
            cudaMemcpy(odata, dev_idata, size * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < size; i++) {
                printf("out[%d] = %d : ", i, odata[i]);
            }
            printf("\n");

            cudaFree(dev_idata);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
