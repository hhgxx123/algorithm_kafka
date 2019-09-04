#include "utils.h"
   
int get_milliseconds()
{
	struct timeb t;
	ftime(&t);
	return t.millitm;
}

const char *get_loacal_time(int milliseconds) {
	static char date[100];

    time_t tim_t;
    struct tm loc_t;
    
    std::time(&tim_t);
    loc_t = *localtime(&tim_t);

    memset(date, 0, sizeof(date));
    if(milliseconds < 0) {
        sprintf(date, "%04d-%02d-%02dT%02d:%02d:%02d.%04d", 
                loc_t.tm_year+1900, loc_t.tm_mon+1, loc_t.tm_mday, 
                loc_t.tm_hour, loc_t.tm_min, loc_t.tm_sec, get_milliseconds());
    }else {
        static int day = loc_t.tm_mday;
        static int hour = loc_t.tm_hour;
        static int min = loc_t.tm_min;
        static int sec = loc_t.tm_sec;
        static int mill = get_milliseconds();

        mill += milliseconds;

        sec += mill / 1000;
        mill = mill % 1000;

        min += sec / 60;
        sec = sec % 60;

        hour += min / 60;
        min = min % 60;

        day += hour / 24;
        hour = hour % 24;

        int year = loc_t.tm_year+1900;
        int mon = loc_t.tm_mon+1;


        sprintf(date, "%04d-%02d-%02dT%02d:%02d:%02d.%04d", 
                year, mon, day, hour, min, sec, mill);
        
    }
    return date;
}

int region_check(int width, int height, cv::Rect src_region, cv::Rect &dst_region) {
    int x1=src_region.x;
    int x2=src_region.x+src_region.width;
    int y1=src_region.y;
    int y2=src_region.y+src_region.height;
    if (x1<0) {
        x1=0;
    }
    if (x2>=width) {
        x2=width-1;
    }
    if(y1<0) {
        y1=0;
    }
    if (y2>=height) {
        y2=height-1;
    }
    if (x1>=x2) {
        return -1;
    }
    if (y1>=y2) {
        return -1;
    }
    dst_region.x=x1;
    dst_region.y=y1;
    dst_region.width=x2-x1;
    dst_region.height=y2-y1;
    return 0;
}



int draw_rect2image(Record *record, int type, cv::Mat &image) {
    if(type == 1) {
        image = record->image;
    }else if(type == 2) {
        if(get_cpuImage<AvsGpuMat>(record->gpu_image, image) < 0)
            return -1;
    }

    for(int i=0; i<record->object_list.size(); i++){
        cv::rectangle(image, 
                cv::Rect(record->object_list[i].region),
                cv::Scalar(255,0,200), 1, 1, 0);
    }
    return 0;
}

int save_record_image(Record *record, int type, cv::Mat &image, std::vector<cv::Point> &line) {
    static int come_num = 0;
    static int go_num = 0;
    come_num += record->come_num;
    go_num += record->go_num;
    cv::putText(image,
            "come_num:"+std::to_string(come_num)
            +"  go_num:"+std::to_string(go_num),
            cv::Point(1920-400, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::line(image, line[0], line[1],
                cv::Scalar(0,0,255), 2, 1, 0);
    for(int i=0; i<record->object_list.size(); i++){
        cv::rectangle(image, 
                cv::Rect(record->object_list[i].region),
                cv::Scalar(0,0,200), 1, 1, 0);
        LOG(INFO) << "algorithm.cpp object_id:" << record->object_list[i].object_id;
        cv::putText(image,
                    "I:" + std::to_string(record->object_list[i].object_id%100000),
                    cv::Point(record->object_list[i].region.x,
                            record->object_list[i].region.y+30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        if(record->object_list[i].touch_line_flag) {
            // if(record->object_list[i].first_touch_line) {
            //     cv::Mat first_image = image(record->object_list[i].region);
            //     cv::imwrite(std::to_string(record->object_list[i].object_id)+".jpg", first_image);
            // }
        // if(record->object_list[i].label == 3 && record->object_list[i].touch_line_flag) {
            cv::Point points[1][4];
            points[0][0] = cv::Point(record->object_list[i].region.x, record->object_list[i].region.y);
            points[0][1] = cv::Point(record->object_list[i].region.x+record->object_list[i].region.width, record->object_list[i].region.y);
            points[0][2] = cv::Point(record->object_list[i].region.x+record->object_list[i].region.width, record->object_list[i].region.y+record->object_list[i].region.height);
            points[0][3] = cv::Point(record->object_list[i].region.x, record->object_list[i].region.y+record->object_list[i].region.height);
            const cv::Point * ppt[1] = {points[0]};
            int npt[] = {4};
            cv::Mat src = image.clone();
            cv::fillPoly(src, ppt, npt, 1, cv::Scalar(255, 0, 255));
            cv::addWeighted(image, 0.7, src, 0.3, 0, image);
   
            cv::putText(image, "first:" + std::to_string(record->object_list[i].first_touch_line),
                    cv::Point(record->object_list[i].region.x,
                            record->object_list[i].region.y+int(record->object_list[i].region.height/2)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 3);
        }    
        
    }
    // m_video_writer.write(image);
    // END
}

int time_map2txt(std::map<std::string, Module_time> *time_map, std::ofstream &out_file) {
    std::map<std::string, Module_time>::iterator obj;

    std::ostringstream line;
    line << "***********/" ;
    for(obj=time_map->begin(); obj!=time_map->end(); ++obj) {
        line << obj->first << ":[";
        if(obj->second.total >= 0) {
            line << "total:" << obj->second.total << ",";
        }
        if(obj->second.pull >= 0) {
            line << "pull:" << obj->second.pull << ",";
        }
        if(obj->second.inference >= 0) {
            line << "inference:" << obj->second.inference << ",";
        }
        if(obj->second.push >= 0) {
            line << "push:" << obj->second.push << ",";
        }
        if(obj->second.gpu_crops.size() > 0) {
            line << "gpu_crop[";
            for(int i=0; i<obj->second.gpu_crops.size(); i++) {
                line << obj->second.gpu_crops[i] << ",";
            }
            line << "],";
        }
        if(obj->second.get_cpus.size() > 0) {
            line << "get_cpu[";
            for(int i=0; i<obj->second.get_cpus.size(); i++) {
                line << obj->second.get_cpus[i] << ",";
            }
            line << "],";
        }
        if(obj->second.cuda_frees.size() > 0) {
            line << "cuda_free[";
            for(int i=0; i<obj->second.cuda_frees.size(); i++) {
                line << obj->second.cuda_frees[i] << ",";
            }
            line << "],";
        }
        if(obj->second.imwrites.size() > 0) {
            line << "imwrite[";
            for(int i=0; i<obj->second.imwrites.size(); i++) {
                line << obj->second.imwrites[i] << ",";
            }
            line << "],";
        }
        if(obj->second.do_jsons.size() > 0) {
            line << "do_json[";
            for(int i=0; i<obj->second.do_jsons.size(); i++) {
                line << obj->second.do_jsons[i] << ",";
            }
            line << "],";
        }
        line << "],\n";
        out_file << line.str();
        line.clear();
    }
    return 0;
}