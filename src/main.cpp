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
#include "utils.h"
#include "gpu_info.h"
#include "decode.h"
#include "grpc.h"

#include "json2kafka.h"

int main(int args, char* argv[]) {
    static GLogHelper gloghelper(argv[0]);

#ifdef _CIF_GRPC
    {
        std::string port = "0.0.0.0:50051";
        Grpc_run(port);
    }
#else
    GpuInfo gpu_info;
    gpu_info.init();
    
    std::map<std::string, int> source;
    source["/nfs-data/wangxy/wangxy/pipeline_data/1.mp4"] = 2;
    source["/nfs-data/wangxy/wangxy/pipeline_data/2.avi"] = 2;
    // source["/nfs-data/yuany/video/2.avi"] = 2;
    // source["/nfs-data/testdata/platform_crossline.mp4"] = 2;
    // source["rtsp://admin:admin123@192.168.10.24"] = 2;
    // source["/home/wangxy/test_code/algorithm_kafa_nvdecode/1.jpg"] = 2;

    //创建线程
 
    
    int type; //type=1,cpu   2,gpu
#ifdef USE_CPU_IMAGE
    type = 1; //type=1,cpu   2,gpu
#else
    type = 2; //type=1,cpu   2,gpu
#endif
    std::map<std::string, int>::iterator tmp;
    {
        int i = 0;
        for(tmp=source.begin(); tmp!=source.end(); ++tmp, ++i) {
            for(int j=0; j<50; j++){
                if(j >= thread_list.size() || 
                        (j < thread_list.size() && thread_map[thread_list[j]] == 0)) {
                    Decode *decode = new Decode();
                    GrpcPram  souceData;

                    std::cout << "tmp->first:" << tmp->first<< std::endl;
                    souceData.source_name = tmp->first;
                    souceData.source_type = tmp->second; //1,rtsp  2,video  3,picture
                    souceData.uuid = std::to_string(i);
                    while((souceData.gpu_id = gpu_info.allot_gpu_index(MIN_GPU_FREEMEM_RATE)) < 0);
                    souceData.valid=1;
                    souceData.line = {cv::Point(300, 600), cv::Point(1220, 600)};
                    // souceData.line = {cv::Point(755, 230), cv::Point(795, 1080)};
                    souceData.lines_interval=5;

                    decode->start_thread(souceData, type);
                    break;
                }
            }
        }
    }
#endif
    for(int i=0; i<gpu_info.get_gpu_num(); i++) {
        LOG(INFO) << "gpu_allot " << i << " : " << gpu_info.get_gpu_allot(i);
    }
    while(1) {
        sleep(10); 
        for(int i=0; i<thread_list.size(); i++){
            if(thread_map[thread_list[i]] == 0) {
                thread_map.erase(thread_list[i]);
                thread_list.erase(thread_list.begin() + i);
            }
        }    
    }
}

