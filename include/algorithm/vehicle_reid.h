#ifndef __VEHICLE_REID__
#define __VEHICLE_REID__

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
// #define VEHICLE_FEATURE_LENGTH 8
#define VEHICLE_FEATURE_LENGTH 2048

using namespace byavs;
class M_VehicleFeature {
    public:
        VehicleFeature m_vehicle_feature;
        float** m_vehicle_feature_data;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record,float ** vehicle_feature);
        int inference(CpuImgBGRArray &images, float ** vehicle_feature);
        int inference(std::vector<GpuMat> &images, float ** vehicle_feature);
        int test(CpuImgBGRArray &images, float ** vehicle_feature);
        int run(Record *record, int type);
        int release();

};

#endif
