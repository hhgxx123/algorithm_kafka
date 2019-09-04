#include "algorithm.h"

 int Algorithm::set_time_map(std::map<std::string, Module_time> *time_map) {
     m_time_map = time_map;


    m_json2kafka.set_time_map(m_time_map);
    m_pedestrian_den.set_time_map(m_time_map);
    m_detector.set_time_map(m_time_map);
    m_tracking.set_time_map(m_time_map);
    m_pedestrian_seg.set_time_map(m_time_map);
    m_cross_line_detect.set_time_map(m_time_map);
    m_face_detector.set_time_map(m_time_map);
    m_keyframe.set_time_map(m_time_map);
    // m_person_reid.set_time_map(m_time_map);
    // m_person_struct.set_time_map(m_time_map);
    // m_vehicle_reid.set_time_map(m_time_map);
    // m_vehicle_struct.set_time_map(m_time_map);
    // m_face_reid.set_time_map(m_time_map);
    // m_mark_detector.set_time_map(m_time_map);
    // m_plate_struct.set_time_map(m_time_map);
 }

int Algorithm::init_FaceReid(int gpu_id){
    std::string model_dir="/data/wuh/project/algorithm_module/data";
    m_face_reid.init(model_dir,gpu_id);;
    LOG_IF(INFO, 1) << "init finished";
}

int Algorithm::inference_FaceReid(cv::Mat &image,std::vector<float> &feature){

        CpuImgBGRArray face_images;
        face_images.push_back(image);
        m_face_reid.inference(face_images,m_face_reid.m_face_feature_data);

        feature.resize(FACE_FEATURE_LENGTH);
        for(int m=0; m<FACE_FEATURE_LENGTH;m++) {
            feature[m] = m_face_reid.m_face_feature_data[0][m];
        }
        return 0;
}


int Algorithm::init_PersonReid(int gpu_id){
    std::string model_dir="/data/wuh/project/algorithm_module/data";
    m_person_reid.init(model_dir,gpu_id);;
    LOG_IF(INFO, 1) << "init finished";
}

int Algorithm::inference_PersonReid(cv::Mat &image,std::vector<float> &feature){

    CpuImgBGRArray face_images;
    face_images.push_back(image);
    m_person_reid.inference(face_images,m_person_reid.m_person_feature_data);

    feature.resize(PED_FEATURE_LENGTH);
    for(int m=0; m<PED_FEATURE_LENGTH;m++) {
        feature[m] = m_person_reid.m_person_feature_data[0][m];
    }
    return 0;
}


int Algorithm::init(const int gpu_id, std::string camera_id, std::string image_path,
        int partition,std::string brokers,std::string topic, 
        std::vector<cv::Point> line, int lines_interval) {
    std::string model_dir="/data/wuh/project/algorithm_module/data";

    m_camera_id = camera_id;

    m_json2kafka.init(partition, brokers, topic, image_path, m_camera_id);

    m_pedestrian_den.init(model_dir,gpu_id);
    m_detector.init(model_dir,gpu_id);
    m_tracking.init(model_dir,gpu_id);
    m_pedestrian_seg.init(model_dir,gpu_id);
    
    m_line = line;
    byavs::CrossLineParas crossline_pars;
    crossline_pars.line.x1 = m_line[0].x;
    crossline_pars.line.y1 = m_line[0].y;
    crossline_pars.line.x2 = m_line[1].x;
    crossline_pars.line.y2 = m_line[1].y;
    crossline_pars.lines_interval = lines_interval;
    m_cross_line_detect.init(model_dir,crossline_pars,gpu_id);
    
    m_face_detector.init(model_dir,gpu_id);
    m_keyframe.init(model_dir,gpu_id);
    m_person_reid.init(model_dir,gpu_id);
    m_person_struct.init(model_dir,gpu_id);
    m_vehicle_reid.init(model_dir,gpu_id);
    m_vehicle_struct.init(model_dir,gpu_id);
    m_face_reid.init(model_dir,gpu_id);
    m_mark_detector.init(model_dir,gpu_id);
    m_plate_struct.init(model_dir,gpu_id);
/*
    m_video_writer.open("crossline_detect.avi", 
                // CV_FOURCC('M', 'J', 'P', 'G'), 6,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 6,
                cv::Size(1920, 1080), true);
    
*/
	LOG_IF(INFO, 1) << "init finished";
}

int Algorithm::inference(Record* record, int type, std::string image_name) {
	LOG_IF(INFO, 1) << "algorithm inference begin!";
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    m_pedestrian_den.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["density"].total = time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    m_detector.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["detector"].total = time_rep;
    
    // time_start = std::chrono::high_resolution_clock::now(); 
    // cv::Mat image;
    // draw_rect2image(record, type, image);
    // LOG_IF(INFO, 1) << "time of draw_rect2image start to draw_rect2image end:" <<time_rep;
    
    time_start = std::chrono::high_resolution_clock::now();
    m_tracking.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["tracking"].total = time_rep;
    
    
    time_start = std::chrono::high_resolution_clock::now(); 
    m_cross_line_detect.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["cross_line"].total = time_rep;

    time_start = std::chrono::high_resolution_clock::now(); 
    m_face_detector.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["face_detector"].total = time_rep;
    LOG_IF(INFO, 1) << "time of face_detector:" <<time_rep;
    
    time_start = std::chrono::high_resolution_clock::now(); 
    m_pedestrian_seg.run(record, type);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["seg"].total = time_rep;

    // time_start = std::chrono::high_resolution_clock::now();
    // save_record_image(record, type, image, m_line);
    // LOG_IF(INFO, 1) << "time of save_record_image start to save_record_image end:" <<time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    m_json2kafka.tracking_push(record, image_name, m_line);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["json_tracking_push"].total = time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    std::vector<Record*> record_list;
    if(type == 1) {
        KeyInputCPUArray key_frame_array;
        KeyOutputCPUArray keyframe_result;
        if (m_keyframe.cpu_pull_data(record, key_frame_array) == 0){
            m_keyframe.inference(key_frame_array, keyframe_result);
            m_keyframe.push_data(record_list, keyframe_result);
        }
    }else if(type == 2){
        KeyInputGPUArray key_frame_array_gpu;
        KeyOutputGPUArray keyframe_result_gpu;
        if (m_keyframe.gpu_pull_data(record, key_frame_array_gpu)==0){
            m_keyframe.inference(key_frame_array_gpu, keyframe_result_gpu);
            m_keyframe.push_data(record_list, keyframe_result_gpu);
            //cudaFree(record->gpu_image.data);
            //record->gpu_image.data=NULL;
            // delete(record);
        }
    }
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["keyframe"].total = time_rep;
    LOG_IF(INFO, 1) << "time of keyframe:" <<time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<record_list.size(); i++) {

        LOG_IF(INFO, 1) << "after keyframe i:" << i;
        Record *k_record = record_list[i];

        m_person_reid.run(k_record, type);
        m_person_struct.run(k_record, type);
        m_vehicle_reid.run(k_record, type);
        m_vehicle_struct.run(k_record, type);
        m_mark_detector.run(k_record, type);
        m_plate_struct.run(k_record, type);
        m_face_reid.run(k_record, type);
        m_json2kafka.feature_push(k_record);

        if(type == 2) {
            cudaFree(k_record->gpu_image.data);
            k_record->gpu_image.data == nullptr;
        }
        delete(k_record);

    }
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
    (*m_time_map)["keyframe_after"].total = time_rep;
    LOG_IF(INFO, 1) << "time of keyframe after:" <<time_rep;

}

int Algorithm::release() {

}

