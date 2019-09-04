#ifndef __JSON2KAFKA_H__
#define __JSON2KAFKA_H__

#include <map>
#include <queue>
#include <thread>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "rapidjson/filewritestream.h"
#include <rapidjson/writer.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "record.h"
#include "kafka.h"
#include "cuda_crop.h"
#include "utils.h"

#include <chrono>

#ifdef USE_CPU_IMAGE
#define IMAGE_TYPE cv::Mat
#else
#define IMAGE_TYPE AvsGpuMat
#endif

class Json2Kafka {
private:
    std::string m_camera_id;
    std::string m_hot_heat_path;
    std::string m_region_image_path;
    std::string m_seg_image_path;
    std::string m_face_image_path;
    std::string m_plate_image_path;
    
    ProducerKafka producekafka;    
    std::map<std::string, Module_time> *m_time_map;
public:
     int set_time_map(std::map<std::string, Module_time> *time_map);

public:
    Json2Kafka();
    Json2Kafka(std::string camera_id);

    int init(int partition, std::string &brokers, std::string &topic,
        std::string &image_path, std::string camera_id);
    int sub_object2json(SubObjectRecord &object, rapidjson::Value &json_object,
            rapidjson::Document::AllocatorType& allocator, std::string &timestamp,
            IMAGE_TYPE &image);
    int object2json(ObjectRecord &object, rapidjson::Value &json_object,
            rapidjson::Document::AllocatorType& allocator, std::string &timestamp,
            IMAGE_TYPE &image);
    int record2json(Record *record, std::string &m_camera_id, std::string &image_path, 
            std::string &r_string, std::vector<cv::Point> &line);

    int tracking_push(Record*, std::string &image_path, std::vector<cv::Point> &line);
    int feature_push(Record*);
};

#endif
