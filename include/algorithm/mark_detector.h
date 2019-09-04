#ifndef __MARK_DETECT_H__
#define __MARK_DETECT_H__


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
#include "utils.h"
#include "cuda_crop.h"

using namespace byavs;
class M_MarkDetector {
    public:
        MarkDetector m_detector;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, MarkObjArray &objects, int type);
        int inference(CpuImgBGRArray &images,  MarkObjArray &objects);
		int inference(std::vector<GpuMat> &gpu_images, MarkObjArray &objects);
        int test(CpuImgBGRArray &images, MarkObjArray &objects);
        int run(Record *record, int type);
        int release();

};



#endif
