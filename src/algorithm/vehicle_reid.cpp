#include "vehicle_reid.h"

int M_VehicleFeature::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    VehFeatureParas vehicle_feature_para;
    m_vehicle_feature.init(model_dir,vehicle_feature_para,gpu_id);
    
    m_vehicle_feature_data=(float**)malloc(MAX_BATCH*sizeof(float*));
    for (int i=0;i<MAX_BATCH;i++) {
        *(m_vehicle_feature_data+i)=(float*)malloc(2048*sizeof(float));
    }

    return 0;

}
int M_VehicleFeature::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    images.resize(0);
    cv::Mat image = record->image;
    for (int i =0;i<record->object_list.size();i++)
    {
        cv::Mat temp;
        if (record->object_list[i].label==1)
        {
            // temp.clone();
            temp = image(record->object_list[i].region);
            images.push_back(temp.clone());
        }
    }
    if (images.size()>0) {
        return 0;
    }else {
        return -1;
    }
}

int M_VehicleFeature::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    gpu_images.resize(0);
    for (int i=0; i<record->object_list.size(); i++)
    {
        if (record->object_list[i].label==1)
        {
            cv::Rect box_ped = cv::Rect(floor(record->object_list[i].region.x), floor(record->object_list[i].region.y), 
                        floor(record->object_list[i].region.width), floor(record->object_list[i].region.height));
            // std::cout<<"box x: "<<floor(record->object_list[k].region.x)<<std::endl;                
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, box_ped, gpu_crop_ped);
            gpu_images.push_back(gpu_crop_ped);             
        }
    }
    if (gpu_images.size()>0) {
        return 0;
    }else {
        return -1;
    }
}

int M_VehicleFeature::inference(CpuImgBGRArray &images, float** vehicle_feature) {
    m_vehicle_feature.inference(images,vehicle_feature);
    std::cout<<"M_VehicleFeature cpu inference is end"<<std::endl;
}
int M_VehicleFeature::inference(std::vector<GpuMat> &gpu_images, float** vehicle_feature) {
    m_vehicle_feature.inference(gpu_images,vehicle_feature);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    std::cout<<"M_VehicleFeature gpu inference is end"<<std::endl;
}

int M_VehicleFeature::push_data(Record *record, float** vehicle_feature) {
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label != 1) {
            continue;
        }
        record->object_list[i].feature.resize(VEHICLE_FEATURE_LENGTH);
        for(int j=0; j<VEHICLE_FEATURE_LENGTH; j++) {
            record->object_list[i].feature[j] = vehicle_feature[k][j];
        }
        k++;
    }
    return 0;
}
int M_VehicleFeature::run(Record *record, int type) {
    if(type == 1) {
        CpuImgBGRArray vehicle_images;
        if (cpu_pull_data(record, vehicle_images)==0) {
            inference(vehicle_images, m_vehicle_feature_data);
            push_data(record, m_vehicle_feature_data);
        }
    }else if(type == 2) {
        std::vector<GpuMat> vehicle_images_gpu;
        if (gpu_pull_data(record, vehicle_images_gpu)==0) {
            inference(vehicle_images_gpu, m_vehicle_feature_data);
            push_data(record, m_vehicle_feature_data);
        }        
    }
}

int M_VehicleFeature::release() {
    m_vehicle_feature.release();
    return 0;
}
