/*
    Include order:
        1. This file's header
        2. \n
        3. C system files
        4. C++ system files
        5. \n
        6. Other libraries' header files
        7. Project's header file
*/

#include "main_par.hpp"

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <nvgraph.h>
#include "imagem.h"

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        std::cout << "ERROR: " << status << std::endl;
        exit(0);
    }
}

bool in_vector(std::vector<int> &v, int e) {
    return std::find(std::begin(v), std::end(v), e) != std::end(v);
}

void setup(imagem *img, std::vector<int> &seeds_fg, std::vector<int> &seeds_bg, Params &p) {

    std::vector<float> weights;
    std::vector<int> srcidxs;
    std::vector<int> destoff;
    int r = img->rows;
    int c = img->cols;

    p.nvertices = img->total_size + 2;
    p.nedges = 2*((c - 1)*r + (r - 1)*c) + seeds_fg.size() + seeds_bg.size();

    int c_destoff = 0;
    destoff.push_back(c_destoff);

    // Iterate over each vertex
    for(int i = 0; i < img->total_size; i++) {

        c_destoff = destoff[i];

        // If vertex is a foreground seed, create a new origin vertex
        // with index img->rows*img->cols and connect to it with weight 0.0
        if(in_vector(seeds_fg, i)) {
            c_destoff++;
            srcidxs.push_back(img->total_size);
            weights.push_back(0.0);
        }

        // If vertex is a background seed, create a new origin vertex
        // with index img->rows*img->cols + 1 and connect to it with weight 0.0
        if(in_vector(seeds_bg, i)) {
            c_destoff++;
            srcidxs.push_back(img->total_size + 1);
            weights.push_back(0.0);
        }
        
        // Check for neighbors in four directions and create connections,
        // with the weight being the difference in color between neighboring vertices

        int up = i - img->cols;
        if(up > 0) {
            c_destoff++;
            double up_cost = get_edge(img, i, up);
            srcidxs.push_back(up);
            weights.push_back(up_cost);
        }

        int down = i + img->cols;
        if(down < img->total_size) {
            c_destoff++;
            double down_cost = get_edge(img, i, down);
            srcidxs.push_back(down);
            weights.push_back(down_cost);
        }

        int right = i + 1;
        if(right < img->total_size) {
            c_destoff++;
            double right_cost = get_edge(img, i, right);
            srcidxs.push_back(right);
            weights.push_back(right_cost);
        }

        int left = i - 1;
        if(left >= 0) {
            c_destoff++;
            double left_cost = get_edge(img, i, left);
            srcidxs.push_back(left);
            weights.push_back(left_cost);
        }

        destoff.push_back(c_destoff);
    }

    destoff.push_back(srcidxs.size());

    // Convert vectors into arrays because nvGRAPH only works with arrays
    p.weights = (float*)malloc(weights.size()*sizeof(float));
    for(int i = 0; i < weights.size(); i++){
        p.weights[i] = weights[i];
    }
    
    p.srcidxs = (int*) malloc(srcidxs.size()*sizeof(int));
    for(int i = 0; i < srcidxs.size(); i++){
        p.srcidxs[i] = srcidxs[i];
    }

    p.destoff = (int*) malloc(destoff.size()*sizeof(int));
    for(int i = 0; i < destoff.size(); i++){
        p.destoff[i] = destoff[i];
    }
}

// https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-sssp-example
void SSSP(imagem* img,
        float* weights,
        int* srcidxs,
        int* destoff,
        const size_t nv,
        const size_t ne,
        int seed,
        float* res_sssp) {

    const size_t nvsets = 2;
    const size_t nesets = 1;

    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    void** vertex_dim;

    vertex_dim = (void**) calloc(nvsets, sizeof(void*));
    vertex_dimT = (cudaDataType_t*) calloc(nvsets, sizeof(cudaDataType_t));
    vertex_dim[0] = (void*) res_sssp;
    vertex_dimT[0] = CUDA_R_32F;
    
    CSC = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    CSC->nvertices = nv;
    CSC->nedges = ne;
    CSC->destination_offsets = destoff;
    CSC->source_indices = srcidxs;

    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr(handle, &graph));
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, nvsets, vertex_dimT));
    check(nvgraphAllocateEdgeData(handle, graph, nesets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights, 0));

    check(nvgraphSssp(handle, graph, 0, &seed, 0));
    check(nvgraphGetVertexData(handle, graph, (void*)res_sssp, 0));
    
    free(CSC);
    free(vertex_dim);
    free(vertex_dimT);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));

    return;
}

int main(int argc, char **argv) {

    /* USER INPUT */
    if (argc < 3) {
        std::cout << "Uso: segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string path_output(argv[2]);
    
    int n_fg, n_bg, x, y;
    
    std::cin >> n_fg >> n_bg;
    
    std::vector<int> seeds_fg;
    std::vector<int> seeds_bg;

    imagem *img = read_pgm(path);
    imagem *img_edge = new_image(img->rows, img->cols);

    for(int i = 0; i < n_fg; i++) {
        std::cin >> x >> y;
        seeds_fg.push_back(y * img_edge->cols + x);
    }

    for(int i = 0; i < n_bg; i++) {
        std::cin >> x >> y;
        seeds_bg.push_back(y * img_edge->cols + x);
    }
    /* END USER INPUT */


    /* TIMING VARIABLES */
    cudaEvent_t start, stop;
    cudaEvent_t graph_start, graph_stop;
    cudaEvent_t sssp_start, sssp_stop;
    cudaEvent_t img_start, img_stop;
    float el_total, el_graph, el_sssp, el_img;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&graph_start);
    cudaEventCreate(&graph_stop);
    cudaEventCreate(&sssp_start);
    cudaEventCreate(&sssp_stop);
    cudaEventCreate(&img_start);
    cudaEventCreate(&img_stop);
    /* END TIMING VARIABLES */



    cudaEventRecord(start);

        /* GRAPH CONSTRUCTION */
        Params p;

        cudaEventRecord(graph_start);
        
            setup(img, seeds_fg, seeds_bg, p);

        cudaEventRecord(graph_stop);
        cudaEventSynchronize(graph_stop);
        cudaEventElapsedTime(&el_graph, graph_start, graph_stop);
        /* END GRAPH CONSTRUCTION */



        /* SSSP */
        float* sssp0 = (float*) malloc(p.nvertices*sizeof(float));
        float* sssp1 = (float*) malloc(p.nvertices*sizeof(float));

        cudaEventRecord(sssp_start);

            SSSP(img, p.weights, p.srcidxs, p.destoff, p.nvertices, p.nedges, img->total_size, sssp0);
            SSSP(img, p.weights, p.srcidxs, p.destoff, p.nvertices, p.nedges, img->total_size + 1, sssp1);
        
        cudaEventRecord(sssp_stop);
        cudaEventSynchronize(sssp_stop);
        cudaEventElapsedTime(&el_sssp, sssp_start, sssp_stop);
        /* END SSSP */



        /* IMAGE GENERATION */
        cudaEventRecord(img_start);

            imagem *saida = new_image(img->rows, img->cols);
            for (int k = 0; k < saida->total_size; k++) {
                if (sssp0[k] > sssp1[k]) {
                    saida->pixels[k] = 0;
                } else {
                    saida->pixels[k] = 255;
                }
            }
            write_pgm(saida, path_output);
        
        cudaEventRecord(img_stop);
        cudaEventSynchronize(img_stop);
        cudaEventElapsedTime(&el_img, img_start, img_stop);
        /* END IMAGE GENERATION */

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&el_total, start, stop);

    std::cerr << "TIMING" << '\n';
    std::cerr << "GRAPH: " << el_graph << " ms\n";
    std::cerr << "SSSP: "  << el_sssp  << " ms\n";
    std::cerr << "IMAGE: " << el_img   << " ms\n";
    std::cerr << "TOTAL: " << el_total << " ms\n";
    
    return 0;
}
