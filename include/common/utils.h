#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <sys/timeb.h>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include "record.h"
#include "cuda_crop.h"

#include <sstream>
#include <fstream>

#include <chrono>
#include <map>

int get_milliseconds();

const char *get_loacal_time(int milliseconds);


int region_check(int width, int height, cv::Rect src_region, cv::Rect &dst_region);
int draw_rect2image(Record *record, int type, cv::Mat &image);
int save_record_image(Record *record, int type, cv::Mat &image, std::vector<cv::Point> &line);



typedef struct {
    int pull = -1;
    int inference = -1;
    int push = -1;
    int total = -1;
    std::vector<int> gpu_crops;
    std::vector<int> get_cpus;
    std::vector<int> cuda_frees;
    std::vector<int> imwrites;
    std::vector<int> do_jsons;
}Module_time;

int time_map2txt(std::map<std::string, Module_time> *time_map, std::ofstream &out_file);

#endif
