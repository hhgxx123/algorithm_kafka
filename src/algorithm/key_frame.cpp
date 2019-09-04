#include "key_frame.h"
int M_KeyFrame::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_KeyFrame::init(std::string &model_dir, const int gpu_id) {
    std::cout << "Hello, World!" << std::endl;
    KeyFrameParas keyframe_param;
    keyframe.init(model_dir,keyframe_param,gpu_id);

    return 0;
}

int M_KeyFrame::cpu_pull_data(Record* record, KeyInputCPUArray &key_frame_array) {

    KeyObjectCPUs key_objects;
    for(int i=0; i < record->object_list.size(); i++){
        KeyObjectCPU key_object;
        key_object.detect_id = record->object_list[i].detect_id;
        key_object.id = record->object_list[i].object_id;
        key_object.camID = record->camera_id;
//         key_object.camID = 0;
        key_object.timestep = record->timestamp;
        key_object.label = record->object_list[i].label;
        key_object.box.topLeftX = record->object_list[i].region.x;
        key_object.box.topLeftY = record->object_list[i].region.y;
        key_object.box.width = record->object_list[i].region.width;
        key_object.box.height = record->object_list[i].region.height;
        key_object.cpuImg = record->image;
        key_object.return_state = record->object_list[i].return_status;
        for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
            KeySubObject sub_object;
            sub_object.box.topLeftX = record->object_list[i].sub_object_list[j].region.x;
            sub_object.box.topLeftY = record->object_list[i].sub_object_list[j].region.y;
            sub_object.box.width = record->object_list[i].sub_object_list[j].region.width;
            sub_object.box.height = record->object_list[i].sub_object_list[j].region.height;
            sub_object.label = record->object_list[i].sub_object_list[j].label;
            sub_object.score = record->object_list[i].sub_object_list[j].score;
            sub_object.detect_id = record->object_list[i].sub_object_list[j].detect_id;
            key_object.sub_objects.push_back(sub_object);
        }
        key_objects.push_back(key_object);
    }
    key_frame_array.push_back(key_objects);
    if (key_frame_array.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_KeyFrame::gpu_pull_data(Record* record, KeyInputGPUArray &key_frame_gpu_array) {
    KeyObjectGPUs key_objects;
    for(int i=0; i < record->object_list.size(); i++){
        KeyObjectGPU key_object;
        key_object.detect_id = record->object_list[i].detect_id;
        key_object.id = record->object_list[i].object_id;
        key_object.camID = record->camera_id;
//         key_object.camID = 0;
        key_object.timestep = record->timestamp;
        key_object.label = record->object_list[i].label;
        key_object.box.topLeftX = record->object_list[i].region.x;
        key_object.box.topLeftY = record->object_list[i].region.y;
        key_object.box.width = record->object_list[i].region.width;
        key_object.box.height = record->object_list[i].region.height;

        GpuMat gpu_temp;
        gpu_temp.channels = record->gpu_image.channels;
        gpu_temp.data = record->gpu_image.data;
        gpu_temp.height = record->gpu_image.height;
        gpu_temp.width = record->gpu_image.width;
        key_object.gpuImg = gpu_temp;
        key_object.return_state = record->object_list[i].return_status;
        for(int j=0; j<record->object_list[i].sub_object_list.size(); j++) {
            KeySubObject sub_object;
            sub_object.box.topLeftX = record->object_list[i].sub_object_list[j].region.x;
            sub_object.box.topLeftY = record->object_list[i].sub_object_list[j].region.y;
            sub_object.box.width = record->object_list[i].sub_object_list[j].region.width;
            sub_object.box.height = record->object_list[i].sub_object_list[j].region.height;
            sub_object.label = record->object_list[i].sub_object_list[j].label;
            sub_object.score = record->object_list[i].sub_object_list[j].score;
            sub_object.detect_id = record->object_list[i].sub_object_list[j].detect_id;
            key_object.sub_objects.push_back(sub_object);
        }
        key_objects.push_back(key_object);
    }
    key_frame_gpu_array.push_back(key_objects);
    if (key_frame_gpu_array.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}

int M_KeyFrame::push_data(std::vector<Record*> &record_list, KeyOutputCPUArray &keyframe_result) {
    record_list.clear();
    // static int number = 0;
    for(int k=0; k<keyframe_result.size(); k++) {
        for(int i=0; i<keyframe_result[k].size(); i++) {
            Record *record = new Record();
            record->image = keyframe_result[k][i].cpuImg;
            record->timestamp = keyframe_result[k][i].timestep;
            record->camera_id = keyframe_result[k][i].camID;

            ObjectRecord object;
            object.detect_id = keyframe_result[k][i].detect_id;
            object.label = keyframe_result[k][i].label;
            object.object_id = keyframe_result[k][i].id;
            object.region=cv::Rect(keyframe_result[k][i].box.topLeftX,
                    keyframe_result[k][i].box.topLeftY,
                    keyframe_result[k][i].box.width,
                    keyframe_result[k][i].box.height);
            // //TODO
            // std::stringstream mark_path;
            // mark_path << "markdetector-cpu" << "/" << std::to_string(number) << "/"+std::to_string(i)+"/";
            // std::string sql = "mkdir -p "+mark_path.str()+";";
            // system( sql.c_str() );
            // cv::Mat cpu_image = record->image(object.region);
            // cv::imwrite(mark_path.str()+std::to_string(i)+
            //             "-"+std::to_string(object.label)+
            //             ".jpg", cpu_image);
            // //END
            for(int j=0; j<keyframe_result[k][i].sub_objects.size(); j++) {
                SubObjectRecord sub_object;
                sub_object.region=cv::Rect(keyframe_result[k][i].sub_objects[j].box.topLeftX,
                        keyframe_result[k][i].sub_objects[j].box.topLeftY,
                        keyframe_result[k][i].sub_objects[j].box.width,
                        keyframe_result[k][i].sub_objects[j].box.height);
                sub_object.label = keyframe_result[k][i].sub_objects[j].label;
                sub_object.score = keyframe_result[k][i].sub_objects[j].score;
                sub_object.detect_id = keyframe_result[k][i].sub_objects[j].detect_id;
                object.sub_object_list.push_back(sub_object);
                
                // //TODO
                // cv::Mat plate_image = cpu_image(sub_object.region);
                // cv::imwrite(mark_path.str()+std::to_string(i)+
                //             "-"+std::to_string(object.label)+
                //             "-"+std::to_string(j)+
                //             "-"+std::to_string(sub_object.label)+
                //             ".jpg", plate_image);
                // //END
            }
            object.return_status = keyframe_result[k][i].return_state;
            object.match_flag = keyframe_result[k][i].match_flag;
            object.score = keyframe_result[k][i].score;
            LOG_IF(INFO, 1) << "keyframe ["<< i << "] label:" 
                << keyframe_result[k][i].label << "   detect_id:" 
                << keyframe_result[k][i].detect_id;
            
            record->object_list.push_back(object);

            record_list.push_back(record);
            // number++;
        }
    }
    return 0;
}
int M_KeyFrame::push_data(std::vector<Record*> &record_list, KeyOutputGPUArray &keyframe_result) {
    record_list.clear();
    for(int k=0; k<keyframe_result.size(); k++) {
        for(int i=0; i<keyframe_result[k].size(); i++) {
            Record *record = new Record();
            AvsGpuMat temp_gpu;
            temp_gpu.channels = keyframe_result[k][i].gpuImg.channels;
            temp_gpu.data = keyframe_result[k][i].gpuImg.data;
            temp_gpu.height = keyframe_result[k][i].gpuImg.height;
            temp_gpu.width = keyframe_result[k][i].gpuImg.width;
            record->gpu_image = temp_gpu;
            record->timestamp = keyframe_result[k][i].timestep;
    //         record->camera_id = keyframe_result[k][i].camID;

            ObjectRecord object;
            object.detect_id = keyframe_result[k][i].detect_id;
            object.label = keyframe_result[k][i].label;
            object.object_id = keyframe_result[k][i].id;
            object.region=cv::Rect(keyframe_result[k][i].box.topLeftX,
                    keyframe_result[k][i].box.topLeftY,
                    keyframe_result[k][i].box.width,
                    keyframe_result[k][i].box.height);
            for(int j=0; j<keyframe_result[k][i].sub_objects.size(); j++) {
                SubObjectRecord sub_object;
                sub_object.region=cv::Rect(keyframe_result[k][i].sub_objects[j].box.topLeftX,
                        keyframe_result[k][i].sub_objects[j].box.topLeftY,
                        keyframe_result[k][i].sub_objects[j].box.width,
                        keyframe_result[k][i].sub_objects[j].box.height);
                sub_object.label = keyframe_result[k][i].sub_objects[j].label;
                sub_object.score = keyframe_result[k][i].sub_objects[j].score;
                sub_object.detect_id = keyframe_result[k][i].sub_objects[j].detect_id;
                object.sub_object_list.push_back(sub_object);
            }
            object.return_status = keyframe_result[k][i].return_state;
            object.match_flag = keyframe_result[k][i].match_flag;
            object.score = keyframe_result[k][i].score;
            LOG_IF(INFO, 1) << "keyframe ["<< i << "] label:" 
                << keyframe_result[k][i].label << "   detect_id:" 
                << keyframe_result[k][i].detect_id;
            
            record->object_list.push_back(object);

            record_list.push_back(record);
        }
    }
    return 0;
}
int M_KeyFrame::inference(KeyInputCPUArray &key_frame_array, 
        KeyOutputCPUArray &keyframe_result) {
    LOG(INFO)<<"keyframe cpu inference";            
    keyframe.inference(key_frame_array, keyframe_result);
    return 0;
}
int M_KeyFrame::inference(KeyInputGPUArray &key_frame_gpu_array, 
        KeyOutputGPUArray &gpu_keyframe_result) {
    LOG(INFO)<<"keyframe gpu inference";
    keyframe.inference(key_frame_gpu_array, gpu_keyframe_result);
    return 0;
}
int M_KeyFrame::run(std::vector<Record*> &record_list, int type) {
//     if(type == 1)
//         KeyInputCPUArray key_frame_array;
//         KeyOutputCPUArray keyframe_result;
//         if (cpu_pull_data(record, key_frame_array) == 0){
//             inference(key_frame_array, keyframe_result);
//             push_data(record_list, keyframe_result);
//         }
//     }else if(type == 2){
//         KeyInputGPUArray key_frame_array_gpu;
//         KeyOutputGPUArray keyframe_result_gpu;
//         if (gpu_pull_data(record, key_frame_array_gpu)==0){
//             inference(key_frame_array_gpu, keyframe_result_gpu);
//             push_data(record_list, keyframe_result_gpu);
//         }
//     }
}

int M_KeyFrame::release() {
    return 0;
}
