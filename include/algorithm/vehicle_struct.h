#ifndef __VEHICLE_STRUCT_H__
#define __VEHICLE_STRUCT_H__

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <cuda_runtime_api.h>

#include "byavs.h"
#include "record.h"
#include "utils.h"
#include "cuda_crop.h"

using namespace byavs;
class M_VehicleStruct {
    public:
        VehicleStructured m_vehicle_structed;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, VehAttrArray &vehOut);
        int inference(CpuImgBGRArray &images, VehAttrArray &vehOut);
        int inference(std::vector<GpuMat> &images, VehAttrArray &vehOut);
        int test(CpuImgBGRArray &images, VehAttrArray &vehOut);
        int run(Record *record, int type);
        int release();

};

#endif
