#include "segmentation.hpp"
#include "segmentationpar.hpp"
#include "segmentationseq.hpp"
#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>

#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;


struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

result_sssp SSSP(imagem *img, std::vector<int>& source) {
    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho> Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    result_sssp res(custos, predecessor);
    
    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] = -1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }

    for(int& s: source) {
        Q.push(custo_caminho(0.0, s));
        predecessor[s] = s;
        custos[s] = 0.0;
    }

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int acima = vertex - img->cols;
        if (acima >= 0) {
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) {
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        int direita = vertex + 1;
        if (direita < img->total_size) {
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        int esquerda = vertex - 1;
        if (esquerda >= 0) {
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }
    
    delete[] analisado;
    
    return res;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);
    
    int n_fg, n_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;
    
    std::vector<int> seeds_fg;
    std::vector<int> seeds_bg;

    for(int i = 0; i < n_fg; i++) {
        std::cin >> x >> y;
        seeds_fg.push_back(y * img->cols + x);
    }

    for(int i = 0; i < n_bg; i++) {
        std::cin >> x >> y;
        seeds_bg.push_back(y * img->cols + x);
    }
    
    result_sssp fg = SSSP(img, seeds_fg);
    result_sssp bg = SSSP(img, seeds_bg);
    
    imagem *saida = new_image(img->rows, img->cols);
    
    for (int k = 0; k < saida->total_size; k++) {
        if (fg.first[k] > bg.first[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    
    write_pgm(saida, path_output);    
    return 0;
}
