#include "mark_detector.h"

int M_MarkDetector::init(std::string &model_dir, const int gpu_id) {
    std::cout << "M_mark init" << std::endl;
    MarkDetectParas detect_paras;
    m_detector.init(model_dir,detect_paras,gpu_id);
    return 0;
}

int M_MarkDetector::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    images.resize(0);
    cv::Mat image = record->image;
    for (int i =0;i<record->object_list.size();i++)
    {
        if (record->object_list[i].label==1)
        {
            // temp.clone();
            cv::Mat temp = image(record->object_list[i].region);
            images.push_back(temp.clone());
        }
    }
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_MarkDetector::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    gpu_images.resize(0);
    LOG(INFO)<<"record->object_list.size: "<<record->object_list.size();
    for (int i =0;i<record->object_list.size();i++)
    {
        LOG(INFO)<<"record->object_list[" << i << "]"
            <<"   label:"<<record->object_list[i].label;
        if (record->object_list[i].label == 1)
        {
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, record->object_list[i].region, gpu_crop_ped);                    
            LOG(INFO)<<"Cuda cropped person images has been completed";
            gpu_images.push_back(gpu_crop_ped);
        }
    }
    if (gpu_images.size()>0) {
        return 0;
    }else {
        return -1;
    }
}

int M_MarkDetector::inference(CpuImgBGRArray &images, MarkObjArray &objects) {
    m_detector.inference(images, objects);
    return 0;
}
int M_MarkDetector::inference(std::vector<GpuMat> &gpu_images, MarkObjArray &objects) {
    m_detector.inference(gpu_images, objects);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    return 0;
}


int M_MarkDetector::push_data(Record *record, MarkObjArray &objects, int type) {
    int k = 0;
    // static int number = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if(record->object_list[i].label != 1) {
            continue;
        }
        for (int j=0; j<objects[k].size(); j++)
        {
            LOG(INFO)<<"objects[" << k << "][" << j << "]"
            <<"   label:"<<objects[k][j].label;
            if(objects[k][j].label == 1) {
                // cv::Mat cpu_image;
                SubObjectRecord object;
                cv::Rect region=cv::Rect(objects[k][j].box.topLeftX,
                        objects[k][j].box.topLeftY,
                        objects[k][j].box.width,
                        objects[k][j].box.height);
                if(type == 1) {
                    cv::Mat tmp = record->image(record->object_list[i].region);//crop ped image?
                    if(region_check(tmp.cols, tmp.rows, region, object.region) < 0) {
                        LOG_IF(WARNING, 1) << "mark_datect ["<< i << "]["<< j <<"  false region";
                        continue;
                    }
                    // cpu_image = tmp(object.region);
                    // cv::imwrite("markdetector-cpu-"+std::to_string(number)+
                    //         "-"+std::to_string(k)+"-"+std::to_string(j)+
                    //         "-"+std::to_string(objects[k][j].label)+
                    //         ".jpg", cpu_image);
                }else if(type == 2) {
                    if(region_check(record->object_list[i].region.width, record->object_list[i].region.height, region, object.region) < 0) {
                        LOG_IF(WARNING, 1) << "mark_datect ["<< i << "]["<< j <<"  false region";
                        continue;
                    }
                    // AvsGpuMat gpu_crop_image;
                    // cropAVSGPUMat<AvsGpuMat>(gpu_crop_ped, object.region, gpu_crop_image);
                    // get_cpuImage<AvsGpuMat>(gpu_crop_image, cpu_image);
                    // cv::imwrite("markdetector-gpu-"+std::to_string(number)+
                    //         "-"+std::to_string(k)+"-"+std::to_string(j)+
                    //         "-"+std::to_string(objects[k][j].label)+
                    //         ".jpg", cpu_image);
                    // cudaFree(gpu_crop_image.data);
                }   
                object.label = 5;
                object.score= objects[k][j].score;
    //             object.detect_id = record->object_list[j].detect_id;
                object.detect_id = j;
                record->object_list[i].sub_object_list.push_back(object);

                // number++;
                break;
            }
        }
        k++;
    }
    return 0;
}
int M_MarkDetector::run(Record *record, int type) {
    MarkObjArray mark_obj_array;
    if(type == 1) {
        CpuImgBGRArray vehicle_images;
        if (cpu_pull_data(record, vehicle_images)==0) {
            inference(vehicle_images, mark_obj_array);
            push_data(record, mark_obj_array, type);
        }
    }else if(type == 2) {
        std::vector<GpuMat> gpu_images;
        if (gpu_pull_data(record, gpu_images)==0) {
            inference(gpu_images, mark_obj_array);
            push_data(record, mark_obj_array, type);
        }
    }

}
int M_MarkDetector::release() {
    m_detector.release();
    return 0;
}

