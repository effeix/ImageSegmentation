#ifndef IMAGESEGMENTATION_INC_SEGMENTATIONSEQ_H
#define IMAGESEGMENTATION_INC_SEGMENTATIONSEQ_H

#include "segmentation.hpp"

class SegmentationSeq : public Segmentation {
public:
    SegmentationSeq();
    virtual void run();    
};

#endif
