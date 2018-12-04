#ifndef IMAGESEGMENTATION_INC_SEGMENTATIONPAR_H
#define IMAGESEGMENTATION_INC_SEGMENTATIONPAR_H

#include "segmentation.hpp"

class SegmentationPar : public Segmentation {
public:
    SegmentationPar();
    virtual void run();    
};

#endif
