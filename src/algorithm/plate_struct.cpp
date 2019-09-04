#include "plate_struct.h"

int M_PlateStruct::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    PlateParas plate_params;
    m_plate_structured.init(model_dir,plate_params,gpu_id);
    return 0;

}
int M_PlateStruct::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label == 1) {
            for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
                if (record->object_list[i].sub_object_list[j].label == 5) {
                    cv::Mat image = record->image(record->object_list[i].region);
                    cv::Mat plate_image = image(record->object_list[i].sub_object_list[j].region);
                    images.push_back(plate_image.clone());
                }
            }
        }
    }
    if (images.size()>0) {
        return 0;
    }else {
        return -1;
    }
}
int M_PlateStruct::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label == 1) {
            for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
                if (record->object_list[i].sub_object_list[j].label == 5) {
                    AvsGpuMat img_gpu;
                    cropAVSGPUMat<AvsGpuMat>(record->gpu_image, record->object_list[i].region, img_gpu);
                    GpuMat image_gpu;
                    cropAVSGPUMat<GpuMat>(img_gpu, record->object_list[i].sub_object_list[j].region, image_gpu);
                    gpu_images.push_back(image_gpu);

                    cudaFree(img_gpu.data);
                }
            }
        }
    }
    if (gpu_images.size()>0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PlateStruct::push_data(Record *record, PlateAttrArray &plate_result_array) {
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label != 1) {
            continue;
        }
        for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
            if (record->object_list[i].sub_object_list[j].label != 5) {
                continue;
            }
            record->object_list[i].sub_object_list[j].attribute_map["color"].index = plate_result_array[k].plateColor;
            record->object_list[i].sub_object_list[j].attribute_map["plate_type"].index = plate_result_array[k].plateCategory;
            record->object_list[i].sub_object_list[j].attribute_map["plate_number"].attribute = plate_result_array[k].plateNumber;
            record->object_list[i].sub_object_list[j].type = 5;
            break;
        }
        k++;
    }
}

int M_PlateStruct::inference(CpuImgBGRArray &images, PlateAttrArray &plate_result_array)
{
    // cv::imwrite("plate.jpg", images[0]);
    m_plate_structured.inference(images,plate_result_array);
    LOG_IF(INFO, 1) <<"plate_struct inference is end";
	return 0;
}
int M_PlateStruct::inference(std::vector<GpuMat> &gpu_images, PlateAttrArray &plate_result_array) {
    m_plate_structured.inference(gpu_images,plate_result_array);
    for(int i=0; i<gpu_images.size(); i++) {
        cudaFree(gpu_images[i].data);
    }
    LOG_IF(INFO, 1) <<"plate_struct inference is end";
	return 0;
}
int M_PlateStruct::run(Record *record, int type) {
    PlateAttrArray plateResult;
    if(type == 1) {
        CpuImgBGRArray plate_images;
        if(cpu_pull_data(record, plate_images) == 0) {
            inference(plate_images, plateResult);
            push_data(record, plateResult);
        }
    }else if(type == 2) {
        std::vector<GpuMat> plate_images;
        if(gpu_pull_data(record, plate_images) == 0) {
            inference(plate_images, plateResult);
            push_data(record, plateResult);
        }
    }
}

int M_PlateStruct::release()
{
    m_plate_structured.release();
	return 0;
}

