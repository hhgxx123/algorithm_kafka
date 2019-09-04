#ifndef __GRPC_H__
#define __GRPC_H__
#include <map>
#include <queue>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <stdio.h>

#include "glog_init.h"
#include "algorithm.h"

#include <pthread.h>
#include "grpc_eye_server.h"
#include "utils.h"
#include "gpu_info.h"
#include "decode.h"

#include "json2kafka.h"

class Grpc {
    private:
        Algorithm algorithm;

    public:
        Grpc();
        int feature2json(std::string reid_name,std::string &img_file,
                std::vector<float> &feature,std::string &str_json);
        int image_FaceReid(std::string &img_file,std::string &str_json,int gpu_id);
        int image_PersonReid(std::string &img_file, std::string &str_json,int gpu_id);
};

static std::mutex faceReid_mutex;
static std::mutex personReid_mutex;
static int faceReid_Flag = 0;
static int personReidFlag = 0;
int pipeLineImgFeature_fun(pipeLineSourceImage request,
        pipelineFeatureResponse *reply);
int pipelineaddSource_fun(pipelineSourceData grpc_source_data,
        pipelineResponse *reply);
int Grpc_run(std::string &port); 

#endif
