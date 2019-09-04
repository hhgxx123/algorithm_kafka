#ifndef __FACE_DETECT_H__
#define __FACE_DETECT_H__


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
//
#include "cuda_crop.h"

using namespace byavs;
class M_FaceDetector {
    public:
        FaceDetector m_detector; 
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_ped_images);
        int push_data(Record *record, FaceObjArray &objects, int type);
        int inference(CpuImgBGRArray &ped_images, FaceObjArray &objects);
		int inference(std::vector<GpuMat> &gpu_ped_images, FaceObjArray &objects);
        int test(CpuImgBGRArray &images, FaceObjArray &objects);
        int run(Record *record, int type);
        int release();

};



#endif
