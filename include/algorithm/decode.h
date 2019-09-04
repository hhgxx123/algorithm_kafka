#ifndef __DECODE_H__
#define __DECODE_H__

#include <map>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

// decode
#include "NvDecoder.h"
#include "FFmpegDemuxer.h"
#include "ColorSpace.h"
#include <time.h>
#include "record.h"
#include <map>

// encode
#include "FFmpegDemuxer.h"
#include "ColorSpace.h"
#include "gh_jpegnpp.h"


#include "utils.h"
#include "gpu_info.h"

#include "glog_init.h"
#include "algorithm.h"

#include "grpc_eye_server.h"


extern std::map<std::thread::id, int> thread_map;
extern std::vector<std::thread::id> thread_list;

class Decode {
private:
    CudaJpegEncode m_g_cudajpegencode;
    std::map<std::string, Module_time> *m_time_map;
    Algorithm algorithm;
public:
    int set_time_map(std::map<std::string, Module_time> *time_map);

private:
    GpuInfo gpu_info;
	std::string m_source_path;
	int m_source_type;
	int m_gpu_id;
	std::string m_camera_id;
    std::vector<cv::Point> m_line;
    int m_lines_interval; 
    std::thread::id m_thread_id;
    std::string m_image_path;
private:
    AvsGpuMat img;
    CUcontext g_cuContext = nullptr;
    int eOutputFormat = 4;
    int nFrame = 0;
    int nVideoBytes = 0, nFrameReturned = 0;
    uint8_t *pVideo = nullptr;
    uint8_t **ppFrame;
    // Gpumat img;
    FFmpegDemuxer* demuxer = nullptr;
    NvDecoder* dec = nullptr;
public:
	int init(GrpcPram algorithm_source);
    int nv_encodeInit(RectSize &rectsize, DataBuffer &databuffer);
    int nv_decodeInit(GrpcPram &argv);
    int nv_decode(GrpcPram argv, int type);
    int cv_decodeInit(GrpcPram &argv);
    int cv_decode(GrpcPram argv, int type);
    int start_thread(GrpcPram &argv, int type);
};
#endif
