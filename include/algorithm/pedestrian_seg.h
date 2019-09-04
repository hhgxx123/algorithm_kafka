#ifndef __PEDESTRIAN_SEG_H__
#define __PEDESTRIAN_SEG_H__


#include <iostream>
#include <chrono>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "byavs.h"
#include "record.h"
#include "cuda_crop.h"
#include "utils.h"

using namespace byavs;
class M_PedestrianSeg{
    public:
        PedestrianSegmentation pedstrianSeg;
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int inference(CpuImgBGRArray &ped_images, Segmentation_result& SegImg);
        int inference(std::vector<GpuMat> &gpu_ped_images, Segmentation_result& SegImg);
        int push_data(Record *record, Segmentation_result& SegImg);
        int run(Record *record, int type);
        int release();

    private:
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
};

#endif
