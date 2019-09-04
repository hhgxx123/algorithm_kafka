#include "pedestrian_den.h"

int M_PedestrianDen::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_PedestrianDen::init(std::string &model_dir, const int gpu_id)
{
    std::cout << "Hello World" << std::endl;
    PedestrianDensityParas pedestriandensity_para;
    pedestrianDen.init(model_dir,pedestriandensity_para,gpu_id);
    return 0;
}

int M_PedestrianDen::cpu_pull_data(Record* record, CpuImgBGRArray &images)
{
    images.push_back(record->image);
    if (images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianDen::gpu_pull_data(Record *record, std::vector<GpuMat> &gpu_images)
{
    GpuMat gpu_mat;
    gpu_mat.channels = record->gpu_image.channels;
    gpu_mat.data = record->gpu_image.data;
    gpu_mat.height = record->gpu_image.height;
    gpu_mat.width = record->gpu_image.width;
    gpu_images.push_back(gpu_mat); 
    if (gpu_images.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_PedestrianDen::inference(CpuImgBGRArray &images, DensityRes& densityRes)
{
    pedestrianDen.inference(images,densityRes);
    std::cout << "M_PedestrianDen cpu inference is end" << std::endl;
    return 0;
}

int M_PedestrianDen::inference(std::vector<GpuMat> &gpu_images, DensityRes& densityRes)
{
    pedestrianDen.inference(gpu_images,densityRes);
    std::cout << "M_PedestrianDen gpu inference is end" << std::endl;
    return 0;
}

int  M_PedestrianDen::push_data(Record *record, DensityRes& densityRes)
{
    for (int i = 0; i < densityRes.size(); i++){
        record->density_number = densityRes[i].persion_num;
        record->hot_heat_image = densityRes[i].heatmap_density;
    }
    return 0;
}
int M_PedestrianDen::run(Record *record, int type) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    DensityRes heat_results;
    if(type == 1) {
        CpuImgBGRArray heatimages;
        if (cpu_pull_data(record, heatimages) == 0) {
            inference(heatimages, heat_results);
            push_data(record, heat_results);
        }
    }else if(type == 2) {
        std::vector<GpuMat> gpu_heatimages;
        time_start = std::chrono::high_resolution_clock::now();
        if (gpu_pull_data(record, gpu_heatimages)==0) {
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["density"].pull = time_rep;

            time_start = std::chrono::high_resolution_clock::now();
            inference(gpu_heatimages, heat_results);
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["density"].inference = time_rep;

            time_start = std::chrono::high_resolution_clock::now();
            push_data(record, heat_results);
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
            (*m_time_map)["density"].push = time_rep;
        }
    }
}
