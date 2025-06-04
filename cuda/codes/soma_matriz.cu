#include <stdio.h>

__global__ void somaMatrizes(int *a, int *b, int *c, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * width + x;

	if (x < width && y < height) 
        c[idx] = a[idx] + b[idx];
}

int main() {
	int width = 16, height = 16;
	int n = width * height;
	size_t size = n * sizeof(int);
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);

	for (int i = 0; i < n; ++i) h_a[i] = h_b[i] = i;

    printf("Matriz h_a/h_b:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", h_a[i * width + j]);
        }
        printf("\n");
    }

    printf("\n");

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((width + 15)/16, (height + 15)/16);
	somaMatrizes<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Matriz h_c:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", h_c[i * width + j]);
        }
        printf("\n");
    }

    printf("\n");

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(h_a); free(h_b); free(h_c);
    
	return 0;
}
