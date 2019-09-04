#include "detect.h"

int M_Detect::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_Detect::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    DetectParas detect_paras;
    detector.init(model_dir,detect_paras,gpu_id);
    return 0;

}
int M_Detect::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    images.push_back(record->image);
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_Detect::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {

    GpuMat avs_gpu_mat;
    avs_gpu_mat.channels = record->gpu_image.channels;
    avs_gpu_mat.data = record->gpu_image.data;
    avs_gpu_mat.height = record->gpu_image.height;
    avs_gpu_mat.width = record->gpu_image.width;
    gpu_images.push_back(avs_gpu_mat); 
    std::cout<<"detect gpu pull data end\n";
    if (gpu_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_Detect::inference(CpuImgBGRArray &images, ObjArray &objects) {
    detector.inference(images,objects);
    std::cout<<"detector cpu inference is end"<<std::endl;
}

int M_Detect::inference(std::vector<GpuMat> &images, ObjArray &objects) {
    detector.inference(images,objects);
    std::cout<<"detector gpu inference is end"<<std::endl;
}

int M_Detect::push_data(Record *record, ObjArray &objects, int type) {
    for(int i=0; i<objects.size(); i++) {
        for(int j=0; j<objects[i].size(); j++)
        {
            ObjectRecord object;
            cv::Rect region;
            region=cv::Rect(objects[i][j].box.topLeftX,
                    objects[i][j].box.topLeftY,
                    objects[i][j].box.width,
                    objects[i][j].box.height);
            if(type == 1) {
                if(region_check(record->image.cols,record->image.rows, region, object.region) < 0) {
                    LOG_IF(INFO, 1) << "datect ["<< i << "]["<< j <<"  false region";
                    continue;
                }
            }else if(type == 2) {
                if(region_check(record->gpu_image.width, record->gpu_image.height, region, object.region) < 0) {
                    LOG_IF(INFO, 1) << "datect ["<< i << "]["<< j <<"  false region";
                    continue;
                }
            }
            std::cout<<"M_Detect: region"<<region<<" object region:"<<object.region<<std::endl;
            LOG_IF(INFO, 1) << "detect ["<< i << "][" << j << "] label:" << objects[i][j].label;
            if (objects[i][j].label==4) {
                objects[i][j].label=2;
            }
            object.label = objects[i][j].label;
            object.score= objects[i][j].score;
            if(object.score < 0.4) {
                continue;
            }
            object.detect_id = j;
            record->object_list.push_back(object);
        }
    }
    LOG(INFO)<<"record->object_list.size: "<<record->object_list.size();
    return 0;
}

int M_Detect::run(Record *record, int type) {
    ObjArray objects;
    if(type ==1) {
        CpuImgBGRArray images;
        if (cpu_pull_data(record, images) == 0) {
            std::cout<<"detect cpu pull data\n";
            inference(images, objects);
            push_data(record, objects, type);
        }
    }else if(type ==2) {
        std::vector<GpuMat> gpu_images;
        if (gpu_pull_data(record, gpu_images) == 0) {
            std::cout<<"detect gpu pull data\n";
            inference(gpu_images, objects);
            push_data(record, objects, type);
        }
    }
}
int M_Detect::release() {
    detector.release();
    return 0;
}
