#ifndef __TRACKING_H__
#define __TRACKING_H__

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
class M_Tracking {
    public:
        Tracking tracking;
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, const int gpu_id);
        int cpu_pull_data(Record* record, TrackeInputCPUArray &deep_sort_cpu_inputs);
        int gpu_pull_data(Record* record, TrackeInputGPUArray &deep_sort_gpu_inputs);        
        int push_data(Record* record, TrackeResultCPUArray &deep_sort_cpu_results);
        int push_data(Record* record, TrackeResultGPUArray &deep_sort_gpu_results);
        int inference(TrackeInputCPUArray &deep_sort_cpu_inputs, 
                TrackeResultCPUArray &deep_sort_cpu_results);
        int inference(TrackeInputGPUArray &deep_sort_gpu_inputs, 
                TrackeResultGPUArray &deep_sort_gpu_results);                
        int run(Record *record, int type);
        int release();

};


#endif
