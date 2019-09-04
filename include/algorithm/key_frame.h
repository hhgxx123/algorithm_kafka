#ifndef __KEY_FRAME_H__
#define __KEY_FRAME_H__

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
class M_KeyFrame {
    public:
        KeyFrame keyframe;
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, KeyInputCPUArray &key_frame_array);
        int gpu_pull_data(Record* record, KeyInputGPUArray &key_frame_gpu_array);
        int push_data(std::vector<Record*> &record_list, KeyOutputCPUArray &results);
        int push_data(std::vector<Record*> &record_list, KeyOutputGPUArray &results);
        int inference(KeyInputCPUArray& inputs, KeyOutputCPUArray& results);
        int inference(KeyInputGPUArray& inputs, KeyOutputGPUArray& results);
        int run(std::vector<Record*> &record_list, int type);
        int release();

};

#endif
