#ifndef _CUDA_CROP_H_
#define _CUDA_CROP_H_

#include "byavs.h"
#include "record.h"
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "Common.h"

template<typename T>
int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, T &gpuMatImg);
// int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, AvsGpuMat &gpuMatImg);

template<typename T>
int get_cpuImage(const T &imgBGR, cv::Mat &cpuImage);
// int get_cpuImage(const byavs::GpuMat &imgBGR, cv::Mat &cpuImage); 


#include "cuda_crop_template.h"
#endif //_CUDA_CROP_H_
