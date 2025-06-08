#include <stdio.h>

__global__ void ola_cuda() {
    printf("Ola mundo da GPU!\n");
}

int main() {
	ola_cuda<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
