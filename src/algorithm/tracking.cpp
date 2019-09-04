#include "tracking.h"

int M_Tracking::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_Tracking::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    byavs::TrackeParas track_param;
    tracking.init(model_dir,track_param,gpu_id);
    return 0;

}
int M_Tracking::cpu_pull_data(Record* record, TrackeInputCPUArray 
        &deep_sort_cpu_inputs) {
    CpuImgBGRArray images;
    images.push_back(record->image.clone());
    deep_sort_cpu_inputs.clear();
    for(int i=0; i<images.size(); i++)
    {
        
        // TrackeInput temp;
        TrackeInputCPU deep_sort_cpu;
        deep_sort_cpu.camID = record->camera_id;
//         deep_sort_cpu.camID = 0;
        deep_sort_cpu.channelID = 0;
        deep_sort_cpu.cpuImg = images[i];
        for(int j=0; j<record->object_list.size(); j++) {
            DetectObject object;
            object.label = record->object_list[j].label;
            object.score = record->object_list[j].score;
            object.box.topLeftX = record->object_list[j].region.x;
            object.box.topLeftY = record->object_list[j].region.y;
            object.box.width = record->object_list[j].region.width;
            object.box.height = record->object_list[j].region.height;
            deep_sort_cpu.objs.push_back(object);

        }
        deep_sort_cpu_inputs.push_back(deep_sort_cpu);

    }
    if (deep_sort_cpu_inputs.size()>0) {
        return 0;
    }else {
        return -1;
    }
}
int M_Tracking::gpu_pull_data(Record* record, TrackeInputGPUArray &deep_sort_gpu_inputs){
    std::cout<<"M_Tracking gpu_pull_data\n";
    std::vector<GpuMat> gpu_images;

    GpuMat avs_gpu_mat;
    avs_gpu_mat.channels = record->gpu_image.channels;
    avs_gpu_mat.data = record->gpu_image.data;
    avs_gpu_mat.height = record->gpu_image.height;
    avs_gpu_mat.width = record->gpu_image.width;  
    std::cout<<"gpu_image.height: "<<record->gpu_image.width<<std::endl;
    gpu_images.push_back(avs_gpu_mat);
    deep_sort_gpu_inputs.clear();
    std::cout<<"gpu_images.size(): "<<gpu_images.size()<<std::endl;

    for (int i=0; i<gpu_images.size(); i++)
    {
        TrackeInputGPU deep_sort_gpu;
        deep_sort_gpu.camID = record->camera_id;
        deep_sort_gpu.channelID = 0;
        deep_sort_gpu.gpuImg = gpu_images[i];
        for (int j=0; j<record->object_list.size(); j++){
            DetectObject object;
            object.label = record->object_list[j].label;
            object.score = record->object_list[j].score;
            object.box.topLeftX = record->object_list[j].region.x;
            object.box.topLeftY = record->object_list[j].region.y;
            object.box.width = record->object_list[j].region.width;
            object.box.height = record->object_list[j].region.height;
            deep_sort_gpu.objs.push_back(object);
        }
        deep_sort_gpu_inputs.push_back(deep_sort_gpu);
    }
    if (deep_sort_gpu_inputs.size()>0){
        return 0;
    }else{
        return -1;
    }
}

int M_Tracking::push_data(Record* record, TrackeResultCPUArray &objects) {

    record->object_list.clear();  
    for (int i=0; i<objects.size(); i++)
    {
        for(int j=0; j<objects[i].size(); j++) {
            ObjectRecord obj_record;
            cv::Rect rect = cv::Rect(objects[i][j].box.topLeftX,
                    objects[i][j].box.topLeftY,
                    objects[i][j].box.width,
                    objects[i][j].box.height);
            if (region_check(record->image.cols, record->image.rows, rect,obj_record.region)<0)
                continue;
            obj_record.label = objects[i][j].label;
            obj_record.score = objects[i][j].score;
            obj_record.object_id = objects[i][j].id;
            obj_record.return_status = objects[i][j].return_state;
            obj_record.detect_id = j;
            // // TODO
            // cv::Mat picture = record->image(obj_record.region);
            // cv::imwrite("tracking"+std::to_string(obj_record.object_id)+
            //             "-"+std::to_string(obj_record.label)+
            //             "-"+std::to_string(obj_record.score)+
            //             "-"+record->timestamp+
            //             ".jpg", picture);
            // // END
            LOG_IF(INFO, 1) << "tracking ["<< i << "][" << j << "] label:";

            record->object_list.push_back(obj_record);
        }
    }
    return 0;
}
int M_Tracking::push_data(Record* record, TrackeResultGPUArray &objects)
{
    record->object_list.clear();
    for (int i=0; i<objects.size(); i++)
    {
        for (int j=0; j<objects[i].size(); j++){
            ObjectRecord obj_record;
            cv::Rect rect = cv::Rect(objects[i][j].box.topLeftX, 
                                objects[i][j].box.topLeftY,
                                objects[i][j].box.width, 
                                objects[i][j].box.height);
            if(region_check(record->gpu_image.width, record->gpu_image.height, rect, obj_record.region) < 0) {
                    LOG_IF(INFO, 1) << "tracking ["<< i << "]["<< j <<"  false region";
                    continue;     
            }           
            obj_record.label = objects[i][j].label;
            obj_record.score = objects[i][j].score;
            obj_record.object_id = objects[i][j].id;
            obj_record.return_status = objects[i][j].return_state;
            obj_record.detect_id = j;
            LOG_IF(INFO, 1) << "tracking ["<< i << "][" << j << "] label:";

            record->object_list.push_back(obj_record);
        }
    }
    return 0;
}


int M_Tracking::inference(TrackeInputCPUArray &deep_sort_cpu_inputs,
        TrackeResultCPUArray &deep_sort_cpu_results) {
	// static int id = 0;
	if(deep_sort_cpu_inputs.size() > 0) {
		// imwrite("test_track"+std::to_string(id)+".jpg", deep_sort_cpu_inputs[0].cpuImg);
		// id++;
        // std::cout<<"M_Tracking inference start"<<std::endl;
    	tracking.inference(deep_sort_cpu_inputs, deep_sort_cpu_results);
        std::cout<<"M_Tracking cpu inference is end"<<std::endl;
	}
    return 0;
}
int M_Tracking::inference(TrackeInputGPUArray &deep_sort_gpu_inputs,
        TrackeResultGPUArray &deep_sort_gpu_results) {
	// static int id = 0;
	if(deep_sort_gpu_inputs.size() > 0) {
		// imwrite("test_track"+std::to_string(id)+".jpg", deep_sort_cpu_inputs[0].cpuImg);
		// id++;
        // std::cout<<"M_Tracking inference start"<<std::endl;
    	tracking.inference(deep_sort_gpu_inputs, deep_sort_gpu_results);
        std::cout<<"M_Tracking gpu inference is end"<<std::endl;
	}
    return 0;
}

int M_Tracking::run(Record *record, int type) {
    if(type == 1) {
        TrackeInputCPUArray deep_sort_cpu_inputs;
        TrackeResultCPUArray deep_sort_cpu_results;
        if (cpu_pull_data(record,deep_sort_cpu_inputs)==0) {
            inference(deep_sort_cpu_inputs, deep_sort_cpu_results);
            push_data(record, deep_sort_cpu_results);
        }
    }else if(type == 2) {
        TrackeInputGPUArray deep_sort_gpu_inputs;
        TrackeResultGPUArray deep_sort_gpu_results;
        if (gpu_pull_data(record, deep_sort_gpu_inputs) == 0) {
            std::cout<<"M_Tracking gpu_pull_data\n";
            inference(deep_sort_gpu_inputs, deep_sort_gpu_results);
            push_data(record, deep_sort_gpu_results);
        }
    }
}

int M_Tracking::release() {

    return 0;
}
