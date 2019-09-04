#include "face_detect.h"

int M_FaceDetector::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_FaceDetector::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    FaceDetectParas detect_paras;
    m_detector.init(model_dir,detect_paras,gpu_id);
    return 0;
}

int M_FaceDetector::cpu_pull_data(Record* record, CpuImgBGRArray &ped_images) {
    ped_images.resize(0);
    cv::Mat image = record->image;
    for (int i =0;i<record->object_list.size();i++)
    {
        if (record->object_list[i].label==3)
        {
            // temp.clone();
//             std::cout<<"cols:"<<image.cols<<" rows:"<<image.rows<<std::endl;
//             std::cout<<"region:"<<record->object_list[i].region<<std::endl;
            cv::Mat temp = image(record->object_list[i].region);
            ped_images.push_back(temp.clone());
        }
    }
    if (ped_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_FaceDetector::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_ped_images) {
    gpu_ped_images.resize(0);
    // cv::Mat image = record->image;
    LOG(INFO)<<"record->object_list.size: "<<record->object_list.size();
    for (int i =0;i<record->object_list.size();i++)
    {
//         LOG(INFO)<<"record->object_list["<<i<<"]   label:"<<record->object_list[i].label;
        if (record->object_list[i].label == 3)
        {
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, record->object_list[i].region, gpu_crop_ped);                    
            std::cout<<"Cuda cropped person images has been completed\n";
            gpu_ped_images.push_back(gpu_crop_ped);
        }
    }
    if (gpu_ped_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_FaceDetector::inference(CpuImgBGRArray &ped_images, FaceObjArray &objects) {
    m_detector.inference(ped_images, objects);
    std::cout<<"M_FaceDetector gpu inference is end\n";
    return 0;
}
int M_FaceDetector::inference(std::vector<GpuMat> &gpu_ped_images, FaceObjArray &objects){
    m_detector.inference(gpu_ped_images, objects);
    for(int i=0; i<gpu_ped_images.size(); i++) {
        cudaFree(gpu_ped_images[i].data);
    }
    std::cout<<"M_FaceDetector gpu inference is end\n";
    return 0;
}

int M_FaceDetector::push_data(Record *record, FaceObjArray &objects, int type) {
    int k = 0;
    LOG(INFO)<<"objects.size: "<<objects.size();
    for(int i=0; i<record->object_list.size(); i++) {
        if(record->object_list[i].label != 3) {
            continue;
        }
        for (int j=0; j<objects[k].size(); j++)
        {
            SubObjectRecord object;
            cv::Rect region=cv::Rect(objects[k][j].box.topLeftX,
                    objects[k][j].box.topLeftY,
                    objects[k][j].box.width,
                    objects[k][j].box.height);

            if(type == 1) {
                cv::Mat tmp = record->image(record->object_list[i].region);//crop ped image?
                //object.region is face region
                if(region_check(tmp.cols, tmp.rows, region, object.region) < 0) {
                    LOG_IF(WARNING, 1) << "face_datect ["<< i << "]["<< j <<"  false region";
                    continue;
                }
            }else if(type == 2) {
                if(region_check(record->object_list[i].region.width, record->object_list[i].region.height, region, object.region) < 0) {
                    LOG_IF(WARNING, 1) << "face_datect ["<< i << "]["<< j <<"  false region";
                    continue;
                }
            }   

            object.label = 4;
            object.score= objects[k][j].score;
//             object.detect_id = record->object_list[k].detect_id;
            object.detect_id = j;
            record->object_list[i].sub_object_list.push_back(object);
            LOG_IF(INFO, 1) << "face_datect ["<< i << "]["<< j <<"  label:" 
                << object.label << "   detect_id:" << object.detect_id;
            break;
        }
        k++;
    }
    return 0;
}
int M_FaceDetector::run(Record *record, int type) {
    FaceObjArray face_objects;
    if(type == 1) {
        CpuImgBGRArray images;
        if(cpu_pull_data(record, images) == 0) {
            inference(images, face_objects);
            push_data(record, face_objects, type);
        }
    }else if(type == 2) {
        std::vector<GpuMat> gpu_images;
        if (gpu_pull_data(record, gpu_images) == 0) {
            inference(gpu_images, face_objects);
            push_data(record, face_objects, type);
        }
    }

}
int M_FaceDetector::release() {
    m_detector.release();
    return 0;
}

