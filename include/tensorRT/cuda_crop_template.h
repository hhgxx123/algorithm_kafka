#ifndef _CUDA_CROP_TEMPLATE_H_
#define _CUDA_CROP_TEMPLATE_H_

#include "byavs.h"
#include "record.h"
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "Common.h"

#include <chrono>

template<typename T>
int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, T &gpuMatImg)
{
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    byavs::GpuMat gpu_mat;
    gpu_mat.channels = imgBGR.channels;
    gpu_mat.data = imgBGR.data;
    gpu_mat.height = imgBGR.height;
    gpu_mat.width = imgBGR.width;
    unsigned  char *cropImg=nullptr;
    cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels* sizeof(unsigned char));
    
    bdavs::cudaCropImage(gpu_mat.data,gpu_mat.width,gpu_mat.height,gpu_mat.channels,
            cropImg,box.x,box.y,box.width,box.height);
    
    gpuMatImg.data = cropImg;
    gpuMatImg.width = box.width;
    gpuMatImg.height = box.height;
    gpuMatImg.channels  =imgBGR.channels;

    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    LOG_IF(INFO, 1) << "time of cropAVSGPUMat:" << time_rep;
    return 0;
}


template<typename T>
int get_cpuImage(const T &imgBGR, cv::Mat &cpuImage) { 
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    cv::Mat c_img;
    cv::Mat g_img;
    if(imgBGR.channels == 3) {
        cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
                CV_8UC3, (uint8_t*) (imgBGR.data));
        LOG(INFO) << "before download";
        gpuBGRA.download(c_img);
        LOG(INFO) << "after download";
        //LOG_IF(INFO, m_glog_valid)<<"After download";
        if(c_img.empty()){
            LOG_IF(INFO, 1)<<"Image is empty!";
            return -1;
        }
        cpuImage = c_img.clone();
    }else if(imgBGR.channels == 4) {
        cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
                CV_8UC4, (uint8_t*) (imgBGR.data));
        gpuBGRA.download(c_img);
        //LOG_IF(INFO, m_glog_valid)<<"After download";
        if(c_img.empty()){
            LOG_IF(INFO, 1)<<"Image is empty!";
            return -1;
        }
        cv::cvtColor(c_img, g_img, CV_BGRA2BGR);
        cpuImage = g_img.clone();
    }
//     cpuImage = c_img.clone();
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    LOG_IF(INFO, 1) << "time of get_cpuImage:" << time_rep;
    return 0;
}

#endif //_CUDA_CROP_TEMPLATE_H_
