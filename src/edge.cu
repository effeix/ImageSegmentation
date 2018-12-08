#include "edge.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "imagem.h"

/**
 * CUDA architecture is composed of a grid of several processing blocks that contain several threads
 * inside. Each thread process one piece of the data and can be accessed by simple array
 * indexing.
 *
 * Each block is generally composed of 256 threads. An example architecture is below:
 *
 *         Block 0 (blockIdx = 0)
 *         ---------------------------------------------------------
 *         |     0     |     1     |     2     | ... |     255     | 
 *         --------------------------------- -----------------------
 *         |-threadIdx-|
 *         |------------------------blockDim-----------------------|
 *
 * As mentioned above, each thread (index) processes one piece of data. E.G an image processing
 * software would use a thread for each pixel. All the blocks can be accessed as if they were 
 * contiguous in memory, as we were acessing a giant array.
 *
 * To reach a specific thread:
 *   i = blockIdx.x * blockDim.x + threadIdx.x
 *   j = blockIdx.y * blockDim.y + threadIdx.y
 *
 * where blockIdx is the index of the current block, blockDim is the block size and threadIdx
 * is the thread index. Similar to an array: i * n_columns + j
*/

// Based on https://github.com/igordsm/supercomp/blob/master/distributed_memory/18-imagens/edge.c
__global__ void edge_filter(int *in, int *out, int rowStart, int rowEnd, int colStart, int colEnd) {   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int di, dj;
    if(i < rowEnd && j < colEnd) {
        int min = 256;
        int max = 0;
        for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
                if(min>in[di*(colEnd-colStart)+dj]) min = in[di*(colEnd-colStart)+dj];
                if(max<in[di*(colEnd-colStart)+dj]) max = in[di*(colEnd-colStart)+dj]; 
            }
        }
        out[i*(colEnd-colStart)+j] = max-min;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Uso: edgefilter entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string path_output(argv[2]);

    imagem *in = read_pgm(path);
    imagem *out = new_image(in->rows, in->cols);

    thrust::device_vector<int> input(in->pixels, in->pixels + in->total_size);
    thrust::device_vector<int> output(out->pixels, out->pixels + out->total_size);

    dim3 dimGrid(ceil(out->rows/16), ceil(out->cols/16), 1);
    dim3 dimBlock(16, 16, 1);

    edge_filter<<<dimGrid, dimBlock>>>(
        thrust::raw_pointer_cast(input.data()),
        thrust::raw_pointer_cast(output.data()),
        0,
        out->rows,
        0,
        out->cols
    );

    thrust::host_vector<int> cpu(output); // Copy of GPU vector 
    for(int i = 0; i < cpu.size(); i++) {
        out->pixels[i] = cpu[i];
    }

    write_pgm(out, path_output);
}
