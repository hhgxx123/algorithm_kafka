#include "person_reid.h"

int M_PedestrianFeature::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    PedFeatureParas ped;
    std::cout<<"gpu_id:"<<gpu_id<<std::endl;
    m_pedstrian_feaure.init(model_dir,ped,gpu_id);

    m_person_feature_data=(float**)malloc(MAX_BATCH*sizeof(float*));
    for (int i=0;i<MAX_BATCH;i++) {
        *(m_person_feature_data+i)=(float*)malloc(2048*sizeof(float));
    }
    return 0;

}
int M_PedestrianFeature::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    images.resize(0);
    cv::Mat image = record->image;
    for (int i =0;i<record->object_list.size();i++)
    {
        cv::Mat temp;
        if (record->object_list[i].label==3)
        {
            // temp.clone();
            temp = image(record->object_list[i].region);
            images.push_back(temp.clone());
        }
    }
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianFeature::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    gpu_images.resize(0);
    for (int i=0; i<record->object_list.size(); i++)
    {
        if (record->object_list[i].label==3)
        {
            cv::Rect box_ped = cv::Rect(floor(record->object_list[i].region.x), floor(record->object_list[i].region.y), 
                        floor(record->object_list[i].region.width), floor(record->object_list[i].region.height));
            // std::cout<<"box x: "<<floor(record->object_list[k].region.x)<<std::endl;                
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, box_ped, gpu_crop_ped);
            gpu_images.push_back(gpu_crop_ped);   
        }    
    }
    
    if(gpu_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianFeature::inference(CpuImgBGRArray &images, float** ped_feature) {
    m_pedstrian_feaure.inference(images,ped_feature);
    std::cout<<"M_PedestrianFeature cpu inference is end"<<std::endl;
}
int M_PedestrianFeature::inference(std::vector<GpuMat> &gpu_images, float** ped_feature) {
    m_pedstrian_feaure.inference(gpu_images, ped_feature);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    std::cout<<"M_PedestrianFeature gpu inference is end"<<std::endl;
}

int M_PedestrianFeature::push_data(Record *record, float** ped_feature) {
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if(record->object_list[i].label != 3) {
            continue;
        }
        record->object_list[i].feature.resize(PED_FEATURE_LENGTH);
        for(int j=0; j<PED_FEATURE_LENGTH; j++) {
            record->object_list[i].feature[j] = ped_feature[0][k*PED_FEATURE_LENGTH+j];
//                 record->object_list[i].feature[j] = ped_feature[i][j];
        }
        k++;
    }
    return 0;
}
int M_PedestrianFeature::run(Record *record, int type) {
    CpuImgBGRArray obj_images;
    std::vector<GpuMat> obj_images_gpu;    
    if(type == 1) {
        if (cpu_pull_data(record, obj_images)==0) {
            inference(obj_images, m_person_feature_data);
            push_data(record, m_person_feature_data);
        }
    }else if(type == 2) {
        if (gpu_pull_data(record, obj_images_gpu)==0) {
            inference(obj_images_gpu, m_person_feature_data);
            push_data(record, m_person_feature_data);
        }
    }
}

int M_PedestrianFeature::release() {
    m_pedstrian_feaure.release();
    return 0;
}
