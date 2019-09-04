#include "face_reid.h"


int M_FaceReid::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    FaceFeatureParas params;
    m_face_feature.init(model_dir,params,gpu_id);

    m_face_feature_data=(float**)malloc(MAX_BATCH*sizeof(float*));
    for (int i=0;i<MAX_BATCH;i++) {
        *(m_face_feature_data+i)=(float*)malloc(128*sizeof(float));
    }
    return 0;

}
int M_FaceReid::cpu_pull_data(Record* record, CpuImgBGRArray &images) {
	for(int i=0; i<record->object_list.size(); i++) {
		if(record->object_list[i].label == 3) {
            if(record->object_list[i].sub_object_list.size() > 0) {
                cv::Mat image = record->image(record->object_list[i].region);
                cv::Mat face_image = image(record->object_list[i].sub_object_list[0].region);
                images.push_back(face_image.clone());
            }
		}
	}
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_FaceReid::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images) {
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label == 3) {
            if (record->object_list[i].sub_object_list.size() > 0) {
                AvsGpuMat ped_img_gpu;
                cropAVSGPUMat<AvsGpuMat>(record->gpu_image, record->object_list[i].region, ped_img_gpu);
                GpuMat face_image_gpu;
                cropAVSGPUMat<GpuMat>(ped_img_gpu, record->object_list[i].sub_object_list[0].region, face_image_gpu);
                gpu_images.push_back(face_image_gpu);
                
                cudaFree(ped_img_gpu.data);
            }
        }
    }
    if (gpu_images.size() > 0)  {
        return 0;
    }else {
        return -1;
    }
}

int M_FaceReid::inference(CpuImgBGRArray &face_images, float **face_feature) {
    m_face_feature.inference(face_images, face_feature);
    return 0;
}
int M_FaceReid::inference(std::vector<GpuMat> &face_images_gpu, float **face_feature) {
    m_face_feature.inference(face_images_gpu, face_feature);
    for(int i=0; i<face_images_gpu.size(); i++) {
        cudaFree(face_images_gpu[i].data);
    }
    return 0;
}

int M_FaceReid::push_data(Record *record, float** face_feature) {
    int k = 0;
    for(int i=0; i<record->object_list.size(); i++) {
        if (record->object_list[i].label !=3) {
            continue;
        }
        for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
            record->object_list[i].sub_object_list[j].feature.resize(FACE_FEATURE_LENGTH);
            for(int m=0; m<FACE_FEATURE_LENGTH; m++) {
                record->object_list[i].sub_object_list[j].feature[m] = \
                                    face_feature[0][k*FACE_FEATURE_LENGTH+m];
            }
        }
        k++;
    }
    return 0;
}
int M_FaceReid::run(Record *record, int type) {
    if(type == 1) {
        CpuImgBGRArray face_images;
        if(cpu_pull_data(record, face_images) == 0) {
            inference(face_images, m_face_feature_data);
            push_data(record, m_face_feature_data);
        }
    }else if(type ==2) {
        std::vector<GpuMat> face_images_gpu;
        if(gpu_pull_data(record, face_images_gpu) == 0) {
            inference(face_images_gpu, m_face_feature_data);
            push_data(record, m_face_feature_data);
        }
    }
}
