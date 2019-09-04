#ifndef __DETECT_H__
#define __DETECT_H__

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

using namespace byavs;
class M_Detect {
    public:
        Detector detector;    
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int push_data(Record *record, ObjArray &objects, int type);
        int inference(CpuImgBGRArray &images, ObjArray &objects);
        int inference(std::vector<GpuMat> &images, ObjArray &objects);
        int test(CpuImgBGRArray &images, ObjArray &objects);
        int run(Record *record, int type);
        int release();

};

#endif
