#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__


#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

//#include "../src/module/tensorRT/common.h"
#include <chrono>
#include <cuda_runtime_api.h>

// #include "byavs.h"
// #include "record.h"

#include "detect.h"
#include "tracking.h"
#include "pedestrian_seg.h"
#include "pedestrian_den.h"
#include "cross_line.h"
#include "key_frame.h"
#include "person_reid.h"
#include "person_struct.h"
#include "vehicle_reid.h"
#include "vehicle_struct.h"
#include "face_detect.h"
#include "face_reid.h"
#include "mark_detector.h"
#include "plate_struct.h"
#include "json2kafka.h"

#include "cuda_crop.h"
#include <chrono>

#define MAX_BATCH 30

class Algorithm {
private:

    std::string m_camera_id;
	std::string m_image_path;
    std::vector<cv::Point> m_line;
    
    Json2Kafka m_json2kafka;

    M_Detect m_detector;
    M_Tracking m_tracking;
    M_PedestrianSeg m_pedestrian_seg;
    M_PedestrianDen m_pedestrian_den;
    M_CrossLineDetect m_cross_line_detect;
    M_KeyFrame m_keyframe;
    M_PedestrianFeature  m_person_reid;
    M_PersonStruct m_person_struct;
    M_VehicleFeature m_vehicle_reid;
    M_VehicleStruct m_vehicle_struct;
    M_FaceDetector m_face_detector;
    M_FaceReid m_face_reid;
    M_MarkDetector m_mark_detector;
    M_PlateStruct m_plate_struct;
private:
    cv::VideoWriter m_video_writer;
    std::map<std::string, Module_time> *m_time_map;
public:
     int set_time_map(std::map<std::string, Module_time> *time_map);
   

public:
    int init(int gpu_id, std::string camera_id, std::string image_path,
        int partition,std::string brokers,std::string topic, 
        std::vector<cv::Point> line, int lines_interval);
    int inference(Record* record, int type, std::string image_name);
    int release();

public:
    int init_FaceReid(int gpu_id);
    int inference_FaceReid(cv::Mat &image,std::vector<float> &feature);
    int init_PersonReid(int gpu_id);
    int inference_PersonReid(cv::Mat &image,std::vector<float> &feature);
};

#endif
