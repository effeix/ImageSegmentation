#include "imagem.h"
#include <fstream>
#include <string>

imagem *new_image(int rows, int cols) {
    imagem *img = new imagem;
    img->rows = rows;
    img->cols = cols;
    img->total_size = rows * cols;
    img->pixels = new unsigned char[img->total_size];
    return img;

}

imagem *read_pgm(std::string path) {
    std::ifstream inf(path);
    std::string first_line;
    std::getline(inf, first_line);
    if (first_line != "P2") return NULL;
    
    imagem *img = new imagem;
    inf >> img->cols;
    inf >> img->rows;
    int temp;
    inf >> temp;
    img->total_size = img->rows * img->cols;
    img->pixels = new unsigned char[img->total_size];
    
    for (int k = 0; k < img->total_size; k++) {
        int t;
        inf >> t;
        img->pixels[k] = t;
    }
    
    return img;
}

void write_pgm(imagem *img, std::string path) {
    std::ofstream of(path);
    of << "P2\n" << img->cols << " " <<  img->rows << " 255\n";
    for (int k = 0; k < img->total_size; k++) {
        of << ((int) img->pixels[k]) << " ";
    }
}

