#ifndef IMAGESEGMENTATION_MAIN_PAR_H
#define IMAGESEGMENTATION_MAIN_PAR_H

typedef struct params_t {
    unsigned long nvertices;
    unsigned long nedges;
    float* weights;
    int* destoff;
    int* srcidxs;
} Params;

#endif
