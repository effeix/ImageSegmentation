#ifndef IMAGEM_H
#define IMAGEM_H

#include <string>


typedef struct {
    int rows, cols;
    int total_size;
    unsigned char *pixels;
} imagem;


inline int get_edge(imagem *img, int p1, int p2) {
    int px1 = img->pixels[p1];
    int px2 = img->pixels[p2];
    return abs(px1 - px2);
}

imagem *new_image(int rows, int cols);
imagem *read_pgm(std::string path);
void write_pgm(imagem *img, std::string path);

#endif
