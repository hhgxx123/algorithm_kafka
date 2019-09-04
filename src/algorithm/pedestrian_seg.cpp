#include "pedestrian_seg.h"

int M_PedestrianSeg ::set_time_map(std::map<std::string, Module_time> *time_map) {
     m_time_map = time_map;
 }


int M_PedestrianSeg::init(std::string &model_dir, const int gpu_id)
{
    std::cout << "Hello World" << std::endl;
    PedestrianSegmentationParas pedestrainseg_paras;
    pedstrianSeg.init(model_dir, pedestrainseg_paras, gpu_id);
    return 0;
}

int M_PedestrianSeg::cpu_pull_data(Record *record, CpuImgBGRArray &images)
{
    for (int i = 0; i < record->object_list.size(); i++)
    {
        if (record->object_list[i].label == 3)
        {
            cv::Mat image = record->image(record->object_list[i].region);
//             std::stringstream str;
//             str << "f" << i << ".jpg";
//             cv::imwrite("img" + str.str(), image);
            images.push_back(image.clone());
        }
    }
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianSeg::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_ped_images)
{
    // static int number = 0;
    for (int i =0; i<record->object_list.size(); i++)
    {
        if (record->object_list[i].label == 3)
        {
            GpuMat gpu_crop_ped;
            cropAVSGPUMat<GpuMat>(record->gpu_image, record->object_list[i].region, gpu_crop_ped);                    
            std::cout<<"Cuda cropped person images has been completed\n";
            // cv::Mat picture;
            // get_cpuImage<GpuMat>(gpu_crop_ped, picture);
            // cv::imwrite("seg-pull"+std::to_string(number)+".jpg", picture);
            // number++;

            gpu_ped_images.push_back(gpu_crop_ped);
        }
    }

    if (gpu_ped_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianSeg::inference(CpuImgBGRArray &ped_images, Segmentation_result &SegImg)
{
    pedstrianSeg.inference(ped_images, SegImg);
    std::cout << "M_PedestrianSeg cpu inference is end" << std::endl;
    return 0;
}

int M_PedestrianSeg::inference(std::vector<GpuMat> &gpu_ped_images, Segmentation_result &SegImg) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    pedstrianSeg.inference(gpu_ped_images, SegImg);
    std::cout << "M_PedestrianSeg gpu inference is end" << std::endl;
    
    time_start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<gpu_ped_images.size();i++) {
        cudaFree(gpu_ped_images[i].data);
    }
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["seg"].cuda_frees.push_back(time_rep);
    return 0;
}

int M_PedestrianSeg::push_data(Record *record, Segmentation_result &SegImg)
{
    int k = 0;
    for (int i = 0; i < record->object_list.size(); i++)
    {
        if (record->object_list[i].label != 3) {
            continue;
        }

        record->object_list[i].seg_image = SegImg[k].segmentation_image;
        k++;
    }
    return 0;
}
int M_PedestrianSeg::run(Record *record, int type) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    Segmentation_result pedestrian_result;
    if(type == 1) {
        CpuImgBGRArray pedestrian_images;
        if(cpu_pull_data(record, pedestrian_images) == 0) {
            inference(pedestrian_images, pedestrian_result);
            push_data(record, pedestrian_result);
        }
    }else if(type == 2) {
        std::vector<GpuMat> gpu_ped_images;  
        time_start = std::chrono::high_resolution_clock::now();
        if (gpu_pull_data(record, gpu_ped_images) == 0) {
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["seg"].pull = time_rep;
            
            time_start = std::chrono::high_resolution_clock::now();
            inference(gpu_ped_images, pedestrian_result);
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["seg"].inference = time_rep;

            time_start = std::chrono::high_resolution_clock::now();
            push_data(record, pedestrian_result);
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["seg"].push = time_rep;
        }
    }
}

int M_PedestrianSeg::release()
{
    pedstrianSeg.release();
    return 0;
}
