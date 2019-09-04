#include "person_struct.h"

int M_PersonStruct::init(std::string &model_dir, const int gpu_id)
{
    std::cout << "Hello, World!" << std::endl;
    PedParas ped_params;
    m_person_structured.init(model_dir, ped_params, gpu_id);
    return 0;
}
int M_PersonStruct::cpu_pull_data(Record *record, CpuImgBGRArray &images)
{
    for (int i = 0; i < record->object_list.size(); i++)
    {
        if (record->object_list[i].label == 3)
        {
            cv::Mat image = record->image(record->object_list[i].region);
            images.push_back(image.clone());
        }
    }
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_PersonStruct::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images)
{

    for (int i = 0; i < record->object_list.size(); i++)
    {
        if (record->object_list[i].label == 3)
        {
            cv::Rect box_ped = cv::Rect(floor(record->object_list[i].region.x), floor(record->object_list[i].region.y), 
                        floor(record->object_list[i].region.width), floor(record->object_list[i].region.height));
            // std::cout<<"box x: "<<floor(record->object_list[k].region.x)<<std::endl;                
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, box_ped, gpu_crop_ped);
            gpu_images.push_back(gpu_crop_ped); 
        }
    }
    if (gpu_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PersonStruct::push_data(Record *record, PedAttrArray &pedResult) {
    
    LOG_IF(ERROR, record->object_list.size() != pedResult.size()) 
        << "record->object_list.size():" << record->object_list.size()
        << "         pedResult.size():" << pedResult.size();
    assert(record->object_list.size() == pedResult.size());
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if(record->object_list[i].label != 3) {
            continue;
        }
        record->object_list[i].attribute_map["hair_style"].index = pedResult[k].hairstyle;
        record->object_list[i].attribute_map["age_group"].index = pedResult[k].ageGroup;
        record->object_list[i].attribute_map["upper_category"].index = pedResult[k].upperCategory;
        record->object_list[i].attribute_map["upper_texture"].index = pedResult[k].upperTexture;
        record->object_list[i].attribute_map["upper_color"].index = pedResult[k].upperColor;
        record->object_list[i].attribute_map["lower_category"].index = pedResult[k].lowerCategory;
        record->object_list[i].attribute_map["lower_color"].index = pedResult[k].lowerColor;
        record->object_list[i].attribute_map["shoes_category"].index = pedResult[k].shoesCategory;
        record->object_list[i].attribute_map["shoes_color"].index = pedResult[k].shoesColor;
        record->object_list[i].attribute_map["bag_category"].index = pedResult[k].bagCategory;
        record->object_list[i].attribute_map["hold_baby"].index = pedResult[k].holdBaby;
        record->object_list[i].attribute_map["has_hand_items"].index = pedResult[k].hasHandItems;
        record->object_list[i].attribute_map["hand_items"].index = pedResult[k].handItems;
        record->object_list[i].attribute_map["hat_type"].index = pedResult[k].hatType;
        record->object_list[i].attribute_map["hat_color"].index = pedResult[k].hatColor;
        record->object_list[i].attribute_map["orientation"].index = pedResult[k].orientation;
        record->object_list[i].attribute_map["posture"].index = pedResult[k].posture;
        record->object_list[i].attribute_map["racial"].index = pedResult[k].racial;
        record->object_list[i].attribute_map["ped_height"].index = pedResult[k].pedHeight;
        record->object_list[i].attribute_map["has_umbrella"].index = pedResult[k].hasUmbrella;
        record->object_list[i].attribute_map["hold_phone"].index = pedResult[k].holdPhone;
        record->object_list[i].attribute_map["has_scarf"].index = pedResult[k].hasScarf;
        record->object_list[i].attribute_map["gender"].index = pedResult[k].gender;
        record->object_list[i].attribute_map["has_glasses"].index = pedResult[k].hasGlasses;
        record->object_list[i].attribute_map["has_mask"].index = pedResult[k].hasMask;
        record->object_list[i].attribute_map["has_bag"].index = pedResult[k].hasBag;
        record->object_list[i].attribute_map["has_baby"].index = pedResult[k].hasBaby;
        
        record->object_list[i].type = 3;
        k++;
    }
}

int M_PersonStruct::inference(CpuImgBGRArray &images, PedAttrArray &pedResult) {
    m_person_structured.inference(images, pedResult);
    LOG_IF(INFO, 1) << "Person_struct cpu inference is end";
    return 0;
}
int M_PersonStruct::inference(std::vector<GpuMat> &gpu_images, PedAttrArray &pedResult) {
    m_person_structured.inference(gpu_images, pedResult);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    LOG_IF(INFO, 1) << "Person_struct gpu inference is end";
    return 0;
}
int M_PersonStruct::run(Record *record, int type) {
    PedAttrArray pedResult;
    if(type == 1) {
        CpuImgBGRArray s_obj_images;
        if (cpu_pull_data(record, s_obj_images)==0) {
            inference(s_obj_images, pedResult);
            push_data(record, pedResult);
        }
    }else if(type == 2) {
        std::vector<GpuMat> s_obj_images_gpu;
        if (gpu_pull_data(record, s_obj_images_gpu)==0) {
            inference(s_obj_images_gpu, pedResult);
            push_data(record, pedResult);
        }
    }
}

int M_PersonStruct::release()
{
    m_person_structured.release();
    return 0;
}
