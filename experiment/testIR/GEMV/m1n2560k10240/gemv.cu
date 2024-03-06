#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;
    float *h_A = (float *)malloc(bytes_A);
    float *h_x = (float *)malloc(bytes_x);
    float *h_y = (float *)malloc(bytes_y);
    float *h_y1 = (float *)malloc(bytes_y);

    float *d_A;
    float *d_x;
    float *d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    // 生成A的数据
    for (int i = 0; i < M * N; i++)
    {
        h_A[i] = (float)i / N;
    }

    // 生成x的数据
    for (int i = 0; i < N; i++)
    {
        h_x[i] = 1;
    }
    memset(h_y, 0, M * sizeof(float));
    memset(h_y1, 0, M * sizeof(float));

    int nIter = 1000;
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start, 0));
    cublasSgemv(blas_handle, CUBLAS_OP_T,
                N, M, &alpha,
                d_A, N, d_x, 1, &beta, d_y, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float gpu_time = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("GPU execution time for cublasSgemv: %.3f ms\n", gpu_time);
    checkCudaErrors(cudaMemcpy(h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}
