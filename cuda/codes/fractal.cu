#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define OUTFILE "out_julia_normal_cuda.bmp"

__device__ uchar3 compute_julia_pixel(int x, int y, int altura, int largura, float tint_bias)
{
	// "Zoom in" to a pleasing view of the Julia set
	float X_MIN = -1.6, X_MAX = 1.6, Y_MIN = -0.9, Y_MAX = +0.9;
	float float_y = (Y_MAX - Y_MIN) * (float)y / altura + Y_MIN;
	float float_x = (X_MAX - X_MIN) * (float)x / largura + X_MIN;
	// Point that defines the Julia set
	float julia_real = -.79;
	float julia_img = .15;
	// Maximum number of iteration
	int max_iter = 300;
	// Compute the complex series convergence
	float real = float_y, img = float_x;
	int num_iter = max_iter;

	while ((img * img + real * real < 2 * 2) && (num_iter > 0))
	{
		float xtemp = img * img - real * real + julia_real;
		real = 2 * img * real + julia_img;
		img = xtemp;
		num_iter--;
	}

	uchar3 rgb = make_uchar3(200, 100, 100);

	// Paint pixel based on how many iterations were used, using some funky colors
	float color_bias = (float)num_iter / max_iter;
	rgb.x = (num_iter == 0 ? 200 : -500.0 * pow(tint_bias, 1.2) * pow(color_bias, 1.6));
	rgb.y = (num_iter == 0 ? 100 : -255.0 * pow(color_bias, 0.3));
	rgb.z = (num_iter == 0 ? 100 : 255 - 255.0 * pow(tint_bias, 1.2) * pow(color_bias, 3.0));

	return rgb;
}

__global__ void compute_julia_matrix(unsigned char *pixel_array, int altura, int largura)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < largura && y < altura) {
		int idx = (y * largura + x) * 3;
		uchar3 rgb = compute_julia_pixel(x, y, altura, largura, 1.0);
		pixel_array[idx + 0] = rgb.x;
		pixel_array[idx + 1] = rgb.y;
		pixel_array[idx + 2] = rgb.z;
	}
}

int write_bmp_header(FILE *f, int largura, int altura)
{

	// BMP requires that the size of each row in bytes be a multiple of 4
	unsigned int row_size_in_bytes = largura * 3 +
									((largura * 3) % 4 == 0 ? 0 : (4 - (largura * 3) % 4));

	// Define all fields in the bmp header
	char id[3] = "BM";
	unsigned int filesize = 54 + (int)(row_size_in_bytes * altura * sizeof(char));
	short reserved[2] = {0, 0};
	unsigned int offset = 54;

	unsigned int size = 40;
	unsigned short planes = 1;
	unsigned short bits = 24;
	unsigned int compression = 0;
	unsigned int image_size = largura * altura * 3 * sizeof(char);
	int x_res = 0;
	int y_res = 0;
	unsigned int ncolors = 0;
	unsigned int importantcolors = 0;

	// Write the bytes to the file, keeping track of the
	// number of written "objects"
	size_t ret = 0;
	ret += fwrite(id, sizeof(char), 2, f);
	ret += fwrite(&filesize, sizeof(int), 1, f);
	ret += fwrite(reserved, sizeof(short), 2, f);
	ret += fwrite(&offset, sizeof(int), 1, f);
	ret += fwrite(&size, sizeof(int), 1, f);
	ret += fwrite(&largura, sizeof(int), 1, f);
	ret += fwrite(&altura, sizeof(int), 1, f);
	ret += fwrite(&planes, sizeof(short), 1, f);
	ret += fwrite(&bits, sizeof(short), 1, f);
	ret += fwrite(&compression, sizeof(int), 1, f);
	ret += fwrite(&image_size, sizeof(int), 1, f);
	ret += fwrite(&x_res, sizeof(int), 1, f);
	ret += fwrite(&y_res, sizeof(int), 1, f);
	ret += fwrite(&ncolors, sizeof(int), 1, f);
	ret += fwrite(&importantcolors, sizeof(int), 1, f);

	// Success means that we wrote 17 "objects" successfully
	return (ret != 17);
}

int main(int argc, char *argv[])
{
	int n;
	int area = 0, largura = 0, altura = 0;
	FILE *output_file;
	unsigned char *pixel_array, *pixel_array_device;

	if ((argc <= 1) | (atoi(argv[1]) < 1))
	{
		fprintf(stderr, "Entre 'N' como um inteiro positivo! \n");
		return -1;
	}

	n = atoi(argv[1]);
	altura = n;
	largura = 2 * n;
	area = altura * largura * 3;

	printf("Computando linhas de pixel %d até %d, para uma área total de %d\n", 0, n - 1, area);

	cudaMalloc(&pixel_array_device, area);

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid = dim3(
		(largura + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(altura  + threadsPerBlock.y - 1) / threadsPerBlock.y
	);

	// Chama o kernel...
	compute_julia_matrix<<<blocksPerGrid, threadsPerBlock>>>(pixel_array_device, altura, largura);

	// Allocate mem for the pixels array
	pixel_array = (unsigned char *)calloc(area, sizeof(unsigned char));
	cudaMemcpy(pixel_array, pixel_array_device, area, cudaMemcpyDeviceToHost);
	cudaFree(pixel_array_device);

	// escreve o cabeçalho do arquivo
	output_file = fopen(OUTFILE, "w");
	write_bmp_header(output_file, largura, altura);
	// escreve o array no arquivo
	fwrite(pixel_array, sizeof(unsigned char), area, output_file);
	fclose(output_file);
	free(pixel_array);
	return 0;
}