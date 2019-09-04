#include "cuda_crop.h"
/*
int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, byavs::GpuMat &gpuMatImg)
{

    byavs::GpuMat gpu_mat;
    gpu_mat.channels = imgBGR.channels;
    gpu_mat.data = imgBGR.data;
    gpu_mat.height = imgBGR.height;
    gpu_mat.width = imgBGR.width;
    
    cv::cuda::GpuMat gpuBGRA(gpu_mat.height, gpu_mat.width, 
            CV_8UC4, (uint8_t*) (gpu_mat.data));
//     gpuBGRA.data = (uint8_t*) (imgBGR.data);
    cv::Mat c_img;
    cv::Mat g_img;
    gpuBGRA.download(c_img);
    cv::cvtColor(c_img, g_img, CV_BGRA2BGR);
    cv::cuda::GpuMat gpuBGR;
    gpuBGR.upload(g_img);
    LOG(INFO)<< "cropAVSGPUMat  gpuBGR.channels:" <<gpuBGR.channels();

    unsigned  char *cropImg=nullptr;
//     cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels* sizeof(unsigned char));
    cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels);
    
    bdavs::cudaCropImage(gpuBGR.data,g_img.cols,g_img.rows,3,
            cropImg,box.x,box.y,box.width,box.height);
    
    gpuMatImg.data = cropImg;
    gpuMatImg.width = box.width;
    gpuMatImg.height = box.height;
    gpuMatImg.channels  =imgBGR.channels;
    LOG(INFO)<< "cropAVSGPUMat  gpuMatImg.channels:" <<gpuMatImg.channels;
    
    cv::cuda::GpuMat gpuBGRA_(gpuMatImg.height, gpuMatImg.width, 
            CV_8UC4, (uint8_t*) (gpuMatImg.data));
//     gpuBGRA.data = (uint8_t*) (imgBGR.data);
    gpuBGRA_.download(c_img);
    //LOG_IF(INFO, m_glog_valid)<<"After download";
    if(c_img.empty()){
        LOG_IF(INFO, 1)<<"Image is empty!";
        return -1;
    }
    cv::cvtColor(c_img, g_img, CV_BGRA2BGR);
    static int number = 0;
    cv::imwrite(std::to_string(number)+".jpg", g_img);
    number++;
    return 0;
}
*/
// template<typename T>
// int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, T &gpuMatImg)
// {

//     byavs::GpuMat gpu_mat;
//     gpu_mat.channels = imgBGR.channels;
//     gpu_mat.data = imgBGR.data;
//     gpu_mat.height = imgBGR.height;
//     gpu_mat.width = imgBGR.width;
//     unsigned  char *cropImg=nullptr;
//     cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels* sizeof(unsigned char));
    
//     bdavs::cudaCropImage(gpu_mat.data,gpu_mat.width,gpu_mat.height,gpu_mat.channels,
//             cropImg,box.x,box.y,box.width,box.height);
    
//     gpuMatImg.data = cropImg;
//     gpuMatImg.width = box.width;
//     gpuMatImg.height = box.height;
//     gpuMatImg.channels  =imgBGR.channels;
    
//     return 0;
// }
// int cropAVSGPUMat(const AvsGpuMat &imgBGR,cv::Rect box, AvsGpuMat &gpuMatImg)
// {

//     byavs::GpuMat gpu_mat;
//     gpu_mat.channels = imgBGR.channels;
//     gpu_mat.data = imgBGR.data;
//     gpu_mat.height = imgBGR.height;
//     gpu_mat.width = imgBGR.width;
//     unsigned  char *cropImg=nullptr;
//     cudaMalloc((void**)&cropImg, box.width*box.height*gpu_mat.channels* sizeof(unsigned char));
    
//     bdavs::cudaCropImage(gpu_mat.data,gpu_mat.width,gpu_mat.height,gpu_mat.channels,
//             cropImg,box.x,box.y,box.width,box.height);
    
//     gpuMatImg.data = cropImg;
//     gpuMatImg.width = box.width;
//     gpuMatImg.height = box.height;
//     gpuMatImg.channels  =imgBGR.channels;
//     return 0;
// }
// int get_cpuImage(const AvsGpuMat &imgBGR, cv::Mat &cpuImage) { 
//     cv::Mat c_img;
//     cv::Mat g_img;
//     if(imgBGR.channels == 3) {
//         cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
//                 CV_8UC3, (uint8_t*) (imgBGR.data));
//         LOG(INFO) << "before download";
//         gpuBGRA.download(c_img);
//         LOG(INFO) << "after download";
//         //LOG_IF(INFO, m_glog_valid)<<"After download";
//         if(c_img.empty()){
//             LOG_IF(INFO, 1)<<"Image is empty!";
//             return -1;
//         }
//         cpuImage = c_img.clone();
//     }else if(imgBGR.channels == 4) {
//         cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
//                 CV_8UC4, (uint8_t*) (imgBGR.data));
//         gpuBGRA.download(c_img);
//         //LOG_IF(INFO, m_glog_valid)<<"After download";
//         if(c_img.empty()){
//             LOG_IF(INFO, 1)<<"Image is empty!";
//             return -1;
//         }
//         cv::cvtColor(c_img, g_img, CV_BGRA2BGR);
//         cpuImage = g_img.clone();
//     }
// //     cpuImage = c_img.clone();

//     return 0;
// }
// template<typename T>
// int get_cpuImage(const T &imgBGR, cv::Mat &cpuImage) { 
//     cv::Mat c_img;
//     cv::Mat g_img;
//     if(imgBGR.channels == 3) {
//         cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
//                 CV_8UC3, (uint8_t*) (imgBGR.data));
//         LOG(INFO) << "before download";
//         gpuBGRA.download(c_img);
//         LOG(INFO) << "after download";
//         //LOG_IF(INFO, m_glog_valid)<<"After download";
//         if(c_img.empty()){
//             LOG_IF(INFO, 1)<<"Image is empty!";
//             return -1;
//         }
//         cpuImage = c_img.clone();
//     }else if(imgBGR.channels == 4) {
//         cv::cuda::GpuMat gpuBGRA(imgBGR.height, imgBGR.width, 
//                 CV_8UC4, (uint8_t*) (imgBGR.data));
//         gpuBGRA.download(c_img);
//         //LOG_IF(INFO, m_glog_valid)<<"After download";
//         if(c_img.empty()){
//             LOG_IF(INFO, 1)<<"Image is empty!";
//             return -1;
//         }
//         cv::cvtColor(c_img, g_img, CV_BGRA2BGR);
//         cpuImage = g_img.clone();
//     }
// //     cpuImage = c_img.clone();

//     return 0;
// }
