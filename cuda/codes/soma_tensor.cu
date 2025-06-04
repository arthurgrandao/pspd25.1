#include <stdio.h>

__global__ void somaTensores(int *a, int *b, int *c, int w, int h, int d) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = z * w * h + y * w + x;

	if (x < w && y < h && z < d) 
        c[idx] = a[idx] + b[idx];
}

int main() {
	int w = 4, h = 4, d = 4;
	int n = w * h * d;
	size_t size = n * sizeof(int);

	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);

	for (int i = 0; i < n; ++i) h_a[i] = h_b[i] = i;

    printf("Tensor h_a/h_b:\n");
    for (int z = 0; z < d; ++z) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                printf("%d ", h_a[z * w * h + y * w + x]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n");

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(4, 4, 4);
	dim3 blocksPerGrid((w + 3)/4, (h + 3)/4, (d + 3)/4);
	somaTensores<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, w, h, d);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Tensor h_c:\n");
    for (int z = 0; z < d; ++z) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                printf("%d ", h_c[z * w * h + y * w + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(h_a); free(h_b); free(h_c);
    
	return 0;
}
