#ifndef __PLATE_STRUCT_H__
#define __PLATE_STRUCT_H__

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>
#include <cuda_runtime_api.h>
#include <assert.h>

#include "byavs.h"
#include "record.h"
#include "cuda_crop.h"
#include "utils.h"

using namespace byavs;
class M_PlateStruct {
    public:
        PlateStructured m_plate_structured;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, PlateAttrArray &plateResult);
        int inference(CpuImgBGRArray &images, PlateAttrArray &plateResult);
		int inference(std::vector<GpuMat> &gpu_images, PlateAttrArray &plateResult);
        int test(CpuImgBGRArray &images, PlateAttrArray &PlateResult);
        int run(Record *record, int type);
        int release();
};

#endif

