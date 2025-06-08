#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define BMP_HEADER_SIZE 54
#define ALPHA 0.01      //Thermal diffusivity
#define L 0.2           // Length (m) of the square domain
#define DX 0.02         // local_grid spacing in x-direction
#define DY 0.02         // local_grid spacing in y-direction
#define DT 0.0005       // Time step
#define T 1500.0        //Temperature on ºk of the heat source

__global__ void initialize_grid(double *grid, int nx, int ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx && j < ny) {
        if (i == j || i == nx - 1 - j)
            grid[i * ny + j] = T;
        else
            grid[i * ny + j] = 0.0;
    }
}

__global__ void solve_step(double *grid, double *new_grid, int nx, int ny, double r) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        new_grid[i * ny + j] = grid[i * ny + j]
            + r * (grid[(i + 1) * ny + j] + grid[(i - 1) * ny + j] - 2 * grid[i * ny + j])
            + r * (grid[i * ny + j + 1] + grid[i * ny + j - 1] - 2 * grid[i * ny + j]);
    }
}

__global__ void apply_boundary_conditions(double *grid, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nx) {
        grid[0 * ny + idx] = 0.0;
        grid[(nx - 1) * ny + idx] = 0.0;
    }
    if (idx < ny) {
        grid[idx * ny + 0] = 0.0;
        grid[idx * ny + (ny - 1)] = 0.0;
    }
}

// Function to write BMP file header
void write_bmp_header(FILE *file, int width, int height) {
    unsigned char header[BMP_HEADER_SIZE] = { 0 };
    int file_size = BMP_HEADER_SIZE + 3 * width * height;
    
    header[0] = 'B';
    header[1] = 'M';
    header[2] = file_size & 0xFF;
    header[3] = (file_size >> 8) & 0xFF;
    header[4] = (file_size >> 16) & 0xFF;
    header[5] = (file_size >> 24) & 0xFF;
    header[10] = BMP_HEADER_SIZE;

    header[14] = 40;  // Info header size
    header[18] = width & 0xFF;
    header[19] = (width >> 8) & 0xFF;
    header[20] = (width >> 16) & 0xFF;
    header[21] = (width >> 24) & 0xFF;
    header[22] = height & 0xFF;
    header[23] = (height >> 8) & 0xFF;
    header[24] = (height >> 16) & 0xFF;
    header[25] = (height >> 24) & 0xFF;
    header[26] = 1;   // Planes
    header[28] = 24;  // Bits per pixel

    fwrite(header, 1, BMP_HEADER_SIZE, file);
}

void get_color(double value, unsigned char *r, unsigned char *g, unsigned char *b) {
    if (value >= 500.0) {
        *r = 255; *g = 0; *b = 0; // Red
    }
    else if (value >= 100.0) {
        *r = 255; *g = 128; *b = 0; // Orange
    }
    else if (value >= 50.0) {
        *r = 171; *g = 71; *b = 188; // Lilac
    }
    else if (value >= 25) {
        *r = 255; *g = 255; *b = 0; // Yellow
    }
    else if (value >= 1) {
        *r = 0; *g = 0; *b = 255; // Blue
    }
    else if (value >= 0.1) {
        *r = 5; *g = 248; *b = 252; // Cyan
    }
    else {
        *r = 255; *g = 255; *b = 255; // white
    }
}

//Function to write the grid matrix into the file
void write_grid(FILE *file, double *grid, int nx, int ny) {
    int i, j, padding;
    // Write pixel data to BMP file
    for (i = nx - 1; i >= 0; i--) { // BMP format stores pixels bottom-to-top
        for (j = 0; j < ny; j++) {
            unsigned char r, g, b;
            get_color(grid[i * ny + j], &r, &g, &b);
            fwrite(&b, 1, 1, file); // Write blue channel
            fwrite(&g, 1, 1, file); // Write green channel
            fwrite(&r, 1, 1, file); // Write red channel
        }
        // Row padding for 4-byte alignment (if necessary)
        for (padding = 0; padding < (4 - (nx * 3) % 4) % 4; padding++) {
            fputc(0, file);
        }
    }
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 6) {
        printf("Usage: ./heat_cuda size steps output.bmp threads_x threads_y\n");
        return 1;
    }

    clock_t time_begin, time_end;
    int nx = atoi(argv[1]);
    int ny = nx;
    int steps = atoi(argv[2]);
    double r = ALPHA * DT / (DX * DY);

    size_t size = nx * ny * sizeof(double);
    double *grid, *new_grid;
    double *d_grid, *d_new_grid;

    time_begin=clock();

    grid = (double *)calloc(nx * ny, sizeof(double));
    new_grid = (double *)calloc(nx * ny, sizeof(double));

    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);

    int threads_x = atoi(argv[4]);
    int threads_y = atoi(argv[5]);
    dim3 threadsPerBlock(threads_x, threads_y);
    dim3 numBlocks((ny + threads_x - 1) / threads_x, (nx + threads_y - 1) / threads_y);

    // Initialize the grid
    initialize_grid<<<numBlocks, threadsPerBlock>>>(d_grid, nx, ny);
    cudaMemcpy(d_new_grid, d_grid, size, cudaMemcpyDeviceToDevice);

    // Solve heat equation
    for (int step = 0; step < steps; step++) {
        solve_step<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid, nx, ny, r);
        apply_boundary_conditions<<<numBlocks, threadsPerBlock>>>(d_new_grid, nx, ny);
        double *temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;
    }

    cudaMemcpy(grid, d_grid, size, cudaMemcpyDeviceToHost);

    FILE *file = fopen(argv[3], "wb");
    if (!file) {
        printf("Error opening the output file.\n");
        return 1;
    }

    write_bmp_header(file, nx, ny);
    write_grid(file, grid, nx, ny);

    fclose(file);

    // Free allocated memory
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    free(grid);
    free(new_grid);

    time_end=clock();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("The Execution Time=%fs with a matrix size of %dx%d and %d steps\n",(time_end-time_begin)/(double)CLOCKS_PER_SEC,nx,nx,steps);

    return 0;
}
