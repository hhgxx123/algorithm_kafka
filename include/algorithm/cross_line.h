#ifndef __CROSS_LINE_H__
#define __CROSS_LINE_H__

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
class M_CrossLineDetect {
    public:
        CrossLineOperation m_crossline_detect;   
        std::map<std::string, Module_time> *m_time_map;
    public:
        int set_time_map(std::map<std::string, Module_time> *time_map);
        
    public:
        int init(std::string &model_dir, CrossLineParas &crossline_pars,
                const int gpu_id);
        int cpu_pull_data(Record* record, CrossLineInputArray &obj_lists);
        int push_data(Record *record, CrossLineOutputArray &output_rets);
        int inference(CrossLineInputArray &obj_lists, CrossLineOutputArray &output_rets);
        int run(Record *record, int type);
        int release();

};

#endif
