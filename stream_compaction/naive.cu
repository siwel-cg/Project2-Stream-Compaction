#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernScanNaive(int n, int twoPowD1, int* odata, int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= twoPowD1) {
                odata[index] = idata[index - twoPowD1] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int* dev_idata1;
            int* dev_idata2;
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

            cudaMalloc((void**)&dev_idata1, size * sizeof(int));
            cudaMemcpy(dev_idata1, padded, size * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_idata2, size * sizeof(int));
            cudaMemcpy(dev_idata1, padded, size * sizeof(int), cudaMemcpyHostToDevice);

            

            int blockSize = 64;
            int gridSize = (size + blockSize - 1) / blockSize;
            int npow2;
            for (int d = 0; d < pow2N; d++) {
                npow2 = 1 << d;
                //printf("d = %d : npow2 = %d \n", d, npow2);
                kernScanNaive << < gridSize, blockSize >> > (size, npow2, dev_idata2, dev_idata1);
                std::swap(dev_idata2, dev_idata1);

            }
            
            
            cudaMemcpy(odata, dev_idata1, size * sizeof(int), cudaMemcpyDeviceToHost);

            /**odata = *idata;*/
            /*for (int i = 0; i < size; i++) {
                printf("out[%d] = %d : ", i, odata[i]);
            }
            printf("\n");*/
            cudaFree(dev_idata1);
            cudaFree(dev_idata2);
            timer().endGpuTimer();
        }
    }
}
