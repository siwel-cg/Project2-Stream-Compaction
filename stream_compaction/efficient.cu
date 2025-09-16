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
            iData[index + offDown1] = iData[index] + iData[index + offDown1];

        }

        __global__ void downSweep(int N, int offset, int* iData) {
            int offDown1 = offset >> 1;
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            index *= offset;
            index += offDown1 - 1;
            if (index + offDown1 >= N) {
                return;
            }
            //printf("index = %d, offDown1 = %d, value = %d \n", index, offDown1, iData[index] + iData[index + offDown1]);
            //printf(" |index left: %d, value left: %d, index right: %d, value right: %d, new value = %d | \n", index, iData[index], index + offDown1, iData[index + offDown1], iData[index] + iData[index + offDown1]);
            int temp = iData[index + offDown1];
            iData[index + offDown1] = iData[index] + iData[index + offDown1];
            iData[index] = temp;
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

            cudaMemset(dev_idata + size - 1, 0, sizeof(int));

            for (int d = 0; d < pow2N; d++) { 
                downSweep << <gridSize, blockSize >> > (size, offset, dev_idata);
                offset = offset >> 1;
            }

            cudaMemcpy(odata, dev_idata, size * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 1; i < size; i++) {
                odata[i - 1] = odata[i];
            }

            cudaFree(dev_idata);
            timer().endGpuTimer();
        }

        __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < N) {
                intBuffer[index] = value;
            }
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
            printf("n = %d", n);
            int pow2N = ilog2ceil(n);
            int size = 1 << pow2N;
            int* padded = new int[size];
            for (int i = 0; i < size; i++) {
                if (i >= n) {
                    padded[i] = 0;
                }
                else {
                    padded[i] = idata[i];
                }
            }

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_bool;
            int* host_bool = new int[n];
            int* host_scanResult = new int[n];
            int* dev_indices;
            int* dev_scatter;

            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMalloc((void**)&dev_scatter, n * sizeof(int));

            int blockSize = 64;
            int gridSize = (n + blockSize - 1) / blockSize;

            Common::kernMapToBoolean << <gridSize, blockSize >> > (n, dev_bool, dev_idata);
            cudaMemcpy(host_bool, dev_bool, n * sizeof(int), cudaMemcpyDeviceToHost);
            timer().endGpuTimer();
            scan(n, host_scanResult, host_bool);
            timer().startGpuTimer();

            cudaMemcpy(dev_indices, host_scanResult, n * sizeof(int), cudaMemcpyHostToDevice);
            kernResetIntBuffer << <gridSize, blockSize >> > (n, dev_scatter, 0);
            Common::kernScatter << <gridSize, blockSize >> > (n, dev_scatter, dev_idata, dev_bool, dev_indices);

            for (int i = 0; i < n; i++) {
                printf("bool[%d] = %d : ", i, host_bool[i]);
            }
            printf("\n");
            
            for (int i = 0; i < n; i++) {
               printf("scan[%d] = %d : ", i, host_scanResult[i]);
            }
            printf("\n");

            cudaMemcpy(odata, dev_scatter, n * sizeof(int), cudaMemcpyDeviceToHost);
            int returnNum = 0;
            for (int i = 0; i < n; i++) {
                printf("result[%d] = %d : ", i, odata[i]);
                if (odata[i] == 0) { break; }
                returnNum++;
            }
            printf("\n");


            cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_scatter);

            timer().endGpuTimer();
            return returnNum;
        }
    }
}
