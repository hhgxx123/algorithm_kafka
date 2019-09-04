#include "vehicle_struct.h"


int M_VehicleStruct::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    VehParas params;
    m_vehicle_structed.init(model_dir,params,gpu_id);
    return 0;

}
int M_VehicleStruct::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    images.clear();
    for (int i =0;i<record->object_list.size();i++) {
        if (record->object_list[i].label==1) {
            // temp.clone();
            cv::Mat temp = record->image(record->object_list[i].region);
            images.push_back(temp.clone());
        }
    }
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_VehicleStruct::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    gpu_images.clear();
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
int M_VehicleStruct::inference(CpuImgBGRArray &images, VehAttrArray &vehOut) {
    m_vehicle_structed.inference(images,vehOut);
    return 0;
}
int M_VehicleStruct::inference(std::vector<GpuMat> &gpu_images, VehAttrArray &vehOut) {
    m_vehicle_structed.inference(gpu_images, vehOut);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    return 0;
}

int M_VehicleStruct::push_data(Record *record, VehAttrArray &vehOut) {
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label != 1) {
            continue;
        }
        std::cout<<vehOut[0].vehicleBrand<<std::endl;
            std::cout<<int(vehOut[0].vehicleCategory)<<std::endl;
                std::cout<<int(vehOut[0].vehicleColor)<<std::endl;

        record->object_list[i].attribute_map["brand"].index = vehOut[k].vehicleBrand;
        record->object_list[i].attribute_map["vehicle_type"].index = vehOut[k].vehicleCategory;
        record->object_list[i].attribute_map["color"].index = vehOut[k].vehicleColor;
        
        record->object_list[i].type = 1;      
        k++;
    }
    return 0;
}

int M_VehicleStruct::run(Record *record, int type) {
    VehAttrArray vehOut;
    if(type == 1) {
        CpuImgBGRArray images;
        if (cpu_pull_data(record, images) == 0){
            inference(images, vehOut);
            push_data(record, vehOut);
        }
    }else if(type == 2) {
        std::vector<GpuMat> gpu_images;
        if (gpu_pull_data(record, gpu_images) == 0){
            inference(gpu_images, vehOut);
            push_data(record, vehOut);
        }
    }
}

int M_VehicleStruct::release() {
    m_vehicle_structed.release();
}

