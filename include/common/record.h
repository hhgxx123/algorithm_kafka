#ifndef __RECORD_H__
#define __RECORD_H__


#include <vector>
#include<map>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include <atomic>
#include "byavs.h"

// #include "x2struct/x2struct.hpp"
// #include "rapidjson/document.h"

struct AttributeStruct{
    float score = -1;
    int index = -1;
    int label = -1;
    std::string attribute;
};

struct SubObjectRecord{
   std::map<std::string, AttributeStruct> attribute_map;
    std::vector<float> feature;
    cv::Rect region;
    int label = -1;
    int type = -1;
    float score = -1;
    long long int object_id = -1;//
    int detect_id = -1;//
};
struct ObjectRecord{
    std::map<std::string, AttributeStruct> attribute_map;
    std::vector<float> feature;
    std::vector<float> gait_feature;
    cv::Rect region;
    int return_status;
    int match_flag;
    int label = -1;
    int type = -1;
    float score = -1;
    long long int object_id = -1;//
    int detect_id = -1;//
    cv::Mat seg_image;
    int touch_line_flag;
    int first_touch_line;
    std::vector<SubObjectRecord> sub_object_list;
};
typedef struct
{
    unsigned char* data;
    int height;
    int width;
    int channels;
} AvsGpuMat;

struct Record{
    cv::Mat image;
    std::string camera_id;
    AvsGpuMat gpu_image;
    cv::Mat hot_heat_image;
    int density_number = -1;
    std::map<std::string,std::string> handle_module_map;
    std::vector<ObjectRecord> object_list;
    std::string timestamp;
    int come_num = -1;
    int go_num = -1;
    std::vector<cv::Point> line;
    int lines_interval = -1; 
    std::atomic<int> count_point;
};


#endif
