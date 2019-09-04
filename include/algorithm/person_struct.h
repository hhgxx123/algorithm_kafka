#ifndef __PERSON_STRUCT_H__
#define __PERSON_STRUCT_H__

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
class M_PersonStruct {
    public:
        PersonStructured m_person_structured;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, PedAttrArray &pedResult);
        int inference(CpuImgBGRArray &images, PedAttrArray &pedResult);
		int inference(std::vector<GpuMat> &gpu_images, PedAttrArray &pedResult);
        int test(CpuImgBGRArray &images, ObjArray &pedResult);
        int run(Record *record, int type);
        int release();

};



#endif
