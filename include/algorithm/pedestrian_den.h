#ifndef __PEDESTRIAN_DEN_H__
#define __PEDESTRIAN_DEN_H__

#include <iostream>

#include "byavs.h"
#include "record.h"
#include "utils.h"

using namespace byavs;
class M_PedestrianDen{
    public:
        PedestrianDensity pedestrianDen;
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, CpuImgBGRArray &images);
        int gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images);
        int inference(CpuImgBGRArray &images, DensityRes& densityRes);
        int inference(std::vector<GpuMat> &images, DensityRes& densityRes);
        int push_data(Record *record, DensityRes& densityRes);
        int run(Record *record, int type);
        int release();
};

#endif
