#include <stdio.h>

__global__ void soma_vetores(int *a, int *b, int *c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
        c[idx] = a[idx] + b[idx];
}

int main() {
	int n = 256;
	size_t size = n * sizeof(int);
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);

	for (int i = 0; i < n; ++i) 
        h_a[i] = h_b[i] = i;

    for (int i = 0; i < n; ++i) 
        printf("h_a/h_b[%d] = %d\n", i, h_a[i]); 

    printf("\n");

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	soma_vetores<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) 
        printf("h_c[%d] = %d\n", i, h_c[i]); 

    printf("\n");

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(h_a); free(h_b); free(h_c);

	return 0;
}