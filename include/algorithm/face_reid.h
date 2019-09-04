#ifndef __FACE_REID_H__
#define __FACE_REID_H__

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <cuda_runtime_api.h>

#include "byavs.h"
#include "record.h"
#include "cuda_crop.h"
#include "utils.h"

#define MAX_BATCH 30
#define FACE_FEATURE_LENGTH 128
// #define FACE_FEATURE_LENGTH 8

using namespace byavs;
class M_FaceReid {
    public:
    	FaceFeature m_face_feature;
        float **m_face_feature_data;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, float **face_feature);
        int inference(CpuImgBGRArray &face_images, float **face_feature);
        int inference(std::vector<GpuMat> &face_images_gpu, float **face_feature);
        int test(CpuImgBGRArray &images, float **face_feature);
        int run(Record *record, int type);
        int release();

};

#endif
