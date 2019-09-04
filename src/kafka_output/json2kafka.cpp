#include "json2kafka.h"

Json2Kafka::Json2Kafka(){
}
Json2Kafka::Json2Kafka(std::string camera_id):m_camera_id(camera_id){
}

int Json2Kafka::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int Json2Kafka::init(int partition, std::string &brokers, std::string &topic,
        std::string &image_path, std::string camera_id) {
    m_camera_id = camera_id;
    if(producekafka.init(partition, brokers.c_str(), topic.c_str()) != 0) {
        LOG(ERROR) << "json2kafka.cpp kafka init failed!"
            << "  partition:" << partition
            << "  brokers:" << brokers.c_str()
            << "  topic:" << topic.c_str();
        return -1;
    }

    std::stringstream hot_image_str;
    hot_image_str << image_path << "/" << m_camera_id << "/hot_heat_path/";
	m_hot_heat_path = hot_image_str.str();
	std::string sql = "mkdir -p "+m_hot_heat_path+";";
    system( sql.c_str() );
    std::stringstream region_image_str;
    region_image_str << image_path << "/" << m_camera_id << "/region_image_path/";
	m_region_image_path = region_image_str.str();
	sql = "mkdir -p "+m_region_image_path+";";
    system( sql.c_str() );
    std::stringstream seg_image_str;
    seg_image_str << image_path << "/" << m_camera_id << "/seg_image_path/";
	m_seg_image_path = seg_image_str.str();
	sql = "mkdir -p "+m_seg_image_path+";";
    system( sql.c_str() );
    std::stringstream face_image_str;
    face_image_str << image_path << "/" << m_camera_id << "/face_image_path/";
	m_face_image_path = face_image_str.str();
	sql = "mkdir -p "+m_face_image_path+";";
    system( sql.c_str() );
    std::stringstream plate_image_str;
    plate_image_str << image_path << "/" << m_camera_id << "/plate_image_path/";
	m_plate_image_path = plate_image_str.str();
	sql = "mkdir -p "+m_plate_image_path+";";
    system( sql.c_str() );
}


int Json2Kafka::sub_object2json(SubObjectRecord &object, 
        rapidjson::Value &json_object,
        rapidjson::Document::AllocatorType& allocator, 
        std::string &timestamp, IMAGE_TYPE &image) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();   
    rapidjson::Value region_array(rapidjson::kArrayType);
    region_array.PushBack(object.region.x, allocator);
    region_array.PushBack(object.region.y, allocator);
    region_array.PushBack(object.region.width, allocator);
    region_array.PushBack(object.region.height, allocator);
    json_object.AddMember("region", region_array, allocator);
    
    cv::Mat obj_image;
#ifdef USE_CPU_IMAGE
    obj_image = image(object.region);
#else
    AvsGpuMat gpu_image;
    cropAVSGPUMat<AvsGpuMat>(image, object.region, gpu_image);
    if(get_cpuImage<AvsGpuMat>(gpu_image, obj_image) < 0)
        return -1;
#endif
    if(object.label == 4) {
        std::string face_image_name = m_face_image_path+timestamp+\
                std::to_string(object.detect_id)+".jpg";
        cv::imwrite(face_image_name, obj_image);
        json_object.AddMember("face_image_path", 
                rapidjson::Value().SetString(face_image_name.c_str(), 
                    allocator).Move(),allocator);
    }else if(object.label == 5) {
        std::string plate_image_name = m_plate_image_path+timestamp+\
                std::to_string(object.detect_id)+".jpg";
        cv::imwrite(plate_image_name, obj_image);
        json_object.AddMember("plate_image_path", 
                rapidjson::Value().SetString(plate_image_name.c_str(), 
                    allocator).Move(),allocator);
    }

    json_object.AddMember("label", object.label, allocator);
    json_object.AddMember("object_id", 
            rapidjson::Value().SetString(std::to_string(object.object_id).c_str(), 
                allocator).Move(), allocator);
    if(object.attribute_map.size() > 0) {
        rapidjson::Value attri_map_obj(rapidjson::kObjectType);
        attri_map_obj.AddMember("type", object.type, allocator);
        std::map<std::string, AttributeStruct>::iterator it;
        for(it=object.attribute_map.begin(); 
                it!=object.attribute_map.end(); it++) {
            if(object.label == 5 && it->first == "plate_number") {
                attri_map_obj.AddMember(rapidjson::Value().SetString(it->first.c_str(), 
                        allocator).Move(), 
                        rapidjson::Value().SetString(it->second.attribute.c_str(), 
                            allocator).Move(), allocator);
            }else {
                attri_map_obj.AddMember(rapidjson::Value().SetString(it->first.c_str(), 
                        allocator).Move(), it->second.index, allocator);
            }
        }
        json_object.AddMember("attribute_map", attri_map_obj, allocator);
    }
    if(object.feature.size() > 0) {
        rapidjson::Value feature_array(rapidjson::kArrayType);
        for(int i=0; i<object.feature.size(); i++) {
            feature_array.PushBack(object.feature[i], allocator);
        }
        json_object.AddMember("feature", feature_array, allocator);
    }

#ifndef USE_CPU_IMAGE
    cudaFree(gpu_image.data);
#endif
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["sub_object2json"].total = time_rep;
}
int Json2Kafka::object2json(ObjectRecord &object, 
        rapidjson::Value &json_object,
        rapidjson::Document::AllocatorType& allocator, 
        std::string &timestamp, IMAGE_TYPE &image) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();    
    rapidjson::Value region_array(rapidjson::kArrayType);
    region_array.PushBack(object.region.x, allocator);
    region_array.PushBack(object.region.y, allocator);
    region_array.PushBack(object.region.width, allocator);
    region_array.PushBack(object.region.height, allocator);
    json_object.AddMember("region", region_array, allocator);
    
    std::stringstream region_image_str;
    region_image_str << m_region_image_path << timestamp << object.detect_id << ".jpg";
    std::string region_image_name = region_image_str.str();
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["object2json"].do_jsons.push_back(time_rep);

    time_start = std::chrono::high_resolution_clock::now(); 
    IMAGE_TYPE obj_image;
    cv::Mat cpu_image;
#ifdef USE_CPU_IMAGE
    obj_image = image(object.region);
    cpu_image = obj_image;
#else
    cropAVSGPUMat<IMAGE_TYPE>(image, object.region, obj_image);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["object2json"].gpu_crops.push_back(time_rep);

    time_start = std::chrono::high_resolution_clock::now(); 
    if(get_cpuImage<IMAGE_TYPE>(obj_image, cpu_image) < 0)
        return -1;
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["object2json"].get_cpus.push_back(time_rep);

    time_start = std::chrono::high_resolution_clock::now(); 
#endif
    LOG_IF(INFO, 1) <<"region_image_name:"<< region_image_name;
    cv::imwrite(region_image_name, cpu_image);
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
    (*m_time_map)["object2json"].imwrites.push_back(time_rep);

    time_start = std::chrono::high_resolution_clock::now(); 
    json_object.AddMember("region_image_path", 
                rapidjson::Value().SetString(region_image_name.c_str(), 
                    allocator).Move(),allocator);
    json_object.AddMember("label", object.label, allocator);
    json_object.AddMember("object_id", 
            rapidjson::Value().SetString(std::to_string(object.object_id).c_str(), 
                allocator).Move(), allocator);
    json_object.AddMember("cross_flag", object.touch_line_flag, allocator);
    json_object.AddMember("first_touch", object.first_touch_line, allocator);
    if(object.attribute_map.size() > 0) {
        rapidjson::Value attri_map_obj(rapidjson::kObjectType);
        attri_map_obj.AddMember("type", object.type, allocator);
        std::map<std::string, AttributeStruct>::iterator it;
        for(it=object.attribute_map.begin(); 
                it!=object.attribute_map.end(); it++) {
            attri_map_obj.AddMember(rapidjson::Value().SetString(it->first.c_str(), 
                    allocator).Move(), it->second.index, allocator);
        }
        json_object.AddMember("attribute_map", attri_map_obj, allocator);
    }
    if(object.feature.size() > 0) {
        rapidjson::Value feature_array(rapidjson::kArrayType);
        for(int i=0; i<object.feature.size(); i++) {
            feature_array.PushBack(object.feature[i], allocator);
        }
        json_object.AddMember("feature", feature_array, allocator);
    }
    if(object.label == 1) {
        if(object.sub_object_list.size() > 0) {
            rapidjson::Value plate_obj(rapidjson::kObjectType);
            sub_object2json(object.sub_object_list[0], 
                    plate_obj, allocator, timestamp, obj_image);
            json_object.AddMember("plate", plate_obj, allocator);
        }
    }
    if(object.label == 3) {
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
        (*m_time_map)["object2json"].do_jsons.push_back(time_rep);

        time_start = std::chrono::high_resolution_clock::now(); 
        if(!object.seg_image.empty()) {
            std::string image_name = m_seg_image_path+timestamp+\
                                     std::to_string(object.detect_id)+".jpg";
            LOG_IF(INFO, 1) <<"seg_image_path:"<< image_name;
            cv::imwrite(image_name, object.seg_image);
            time_end = std::chrono::high_resolution_clock::now();
            time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
            (*m_time_map)["object2json"].imwrites.push_back(time_rep);

            time_start = std::chrono::high_resolution_clock::now(); 
            json_object.AddMember("seg_image_path", 
                rapidjson::Value().SetString(image_name.c_str(), 
                    allocator).Move(),allocator);

            rapidjson::Value gait_array(rapidjson::kArrayType);
            for(int j=0; j<5; j++) {
                gait_array.PushBack(j, allocator);
            }
            json_object.AddMember("gait_feature", gait_array, allocator);
        }
        if(object.sub_object_list.size() > 0) {
            rapidjson::Value face_obj(rapidjson::kObjectType);
            sub_object2json(object.sub_object_list[0], 
                    face_obj, allocator, timestamp, obj_image);
            json_object.AddMember("face", face_obj, allocator);
        }
    }
#ifndef USE_CPU_IMAGE
    cudaFree(obj_image.data);
#endif
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    (*m_time_map)["object2json"].total = time_rep;
}
int Json2Kafka::record2json(Record *record, std::string &m_camera_id,
        std::string &image_path, std::string& r_string,
        std::vector<cv::Point> &line) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    IMAGE_TYPE image;
#ifdef USE_CPU_IMAGE
    if(!record->image.empty()) {
        image = record->image;
#else
    if(record->gpu_image.data) {
        image = record->gpu_image;
#endif
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
        (*m_time_map)["record2json"].get_cpus.push_back(time_rep);

        time_start = std::chrono::high_resolution_clock::now(); 
        rapidjson::Document document;
        document.SetObject();
        rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

        
        rapidjson::Value document_obj(rapidjson::kObjectType);

        document_obj.AddMember("image_path",
                rapidjson::Value().SetString(image_path.c_str(), 
                    allocator).Move(),allocator);
        document_obj.AddMember("camera_id",
                rapidjson::Value().SetString(m_camera_id.c_str(), 
                    allocator).Move(),allocator);
        document_obj.AddMember("timestamp", 
                rapidjson::Value().SetString(record->timestamp.c_str(), 
                    allocator).Move(),allocator);


        rapidjson::Value line_arr(rapidjson::kArrayType);
        line_arr.PushBack(line[0].x, allocator).PushBack(line[0].y, allocator);
        line_arr.PushBack(line[1].x, allocator).PushBack(line[1].y, allocator);
        document_obj.AddMember("line", line_arr, allocator);
        document_obj.AddMember("come_num", record->come_num, allocator);
        document_obj.AddMember("go_num", record->go_num, allocator);
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
        (*m_time_map)["record2json"].do_jsons.push_back(time_rep);

        time_start = std::chrono::high_resolution_clock::now(); 
        std::string image_name = m_hot_heat_path+record->timestamp+".jpg";
        LOG_IF(INFO, 1) <<"image_name:"<< image_name;
        cv::imwrite(image_name, record->hot_heat_image);
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
        (*m_time_map)["record2json"].imwrites.push_back(time_rep);

        time_start = std::chrono::high_resolution_clock::now(); 
        document_obj.AddMember("hot_heat_path", 
                rapidjson::Value().SetString(image_name.c_str(), 
                    allocator).Move(),allocator);
//                 rapidjson::StringRef(image_name.c_str()), allocator);
        document_obj.AddMember("density_number", record->density_number, allocator);


        rapidjson::Value vel_json_array(rapidjson::kArrayType);
        rapidjson::Value nvel_json_array(rapidjson::kArrayType);
        rapidjson::Value ped_json_array(rapidjson::kArrayType);
        for(int i=0; i<record->object_list.size(); i++) {
            if(record->object_list[i].label == 1) {
                rapidjson::Value json_obj(rapidjson::kObjectType);
                object2json(record->object_list[i], json_obj, allocator, 
                        record->timestamp, image);
                vel_json_array.PushBack(json_obj, allocator);
            }
            else if(record->object_list[i].label == 2) {
                rapidjson::Value json_obj(rapidjson::kObjectType);
                object2json(record->object_list[i], json_obj, allocator, 
                        record->timestamp, image);
                nvel_json_array.PushBack(json_obj, allocator);
            }
            else if(record->object_list[i].label == 3) {
                rapidjson::Value json_obj(rapidjson::kObjectType);
                object2json(record->object_list[i], json_obj, allocator, 
                        record->timestamp, image);
                ped_json_array.PushBack(json_obj, allocator);
            }
        }
        if(!vel_json_array.Empty() > 0) {
            document_obj.AddMember("vehicle_list", vel_json_array, allocator);
        }
        if(!nvel_json_array.Empty() > 0) {
            document_obj.AddMember("non_vehicle_list", nvel_json_array, allocator);
        }
        if(!ped_json_array.Empty() > 0) {
            document_obj.AddMember("pedestrian_list", ped_json_array, allocator);
        }

        document.AddMember("tracking_data", document_obj, allocator);
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
        (*m_time_map)["record2json"].do_jsons.push_back(time_rep);

        time_start = std::chrono::high_resolution_clock::now(); 
        rapidjson::StringBuffer stringbuf; 
        rapidjson::Writer<rapidjson::StringBuffer> writer(stringbuf);
//         rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(stringbuf);
        writer.SetMaxDecimalPlaces(6);
        document.Accept(writer);
        r_string = stringbuf.GetString();
        time_end = std::chrono::high_resolution_clock::now();
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
        (*m_time_map)["record2json"].do_jsons.push_back(time_rep);
    }
}


int Json2Kafka::tracking_push(Record* record, std::string &image_path, 
        std::vector<cv::Point> &line) {
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::milliseconds::duration::rep time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    std::string r_string;
    record2json(record, m_camera_id, image_path, r_string, line);
    LOG(INFO) << "record_json string_size:"<< r_string.size() 
        << "      size_of:"<< sizeof(r_string) 
        <<"       r_string:"<< r_string;
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    (*m_time_map)["json_tracking_push"].inference = time_rep;

    time_start = std::chrono::high_resolution_clock::now();
    int back_value = producekafka.push_data_to_kafka(r_string.c_str(), strlen(r_string.c_str()));
    time_end = std::chrono::high_resolution_clock::now();
    time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    (*m_time_map)["json_tracking_push"].push = time_rep;
    return back_value;
}

int Json2Kafka::feature_push(Record* record) {
    int back_flag = 0;
    IMAGE_TYPE image;
#ifdef USE_CPU_IMAGE
    if(!record->image.empty()) {
        image = record->image;
#else
    if(record->gpu_image.data) {
        image = record->gpu_image;
#endif
        for(int i=0; i<record->object_list.size(); i++) {
            LOG(INFO) << "feature_push record->object_list.size():" 
                <<record->object_list.size()
                << "     i:" << i << "  label:"<<record->object_list[i].label
                << "     object_id:" << record->object_list[i].object_id
                << "     detect_id:" << record->object_list[i].detect_id;

            rapidjson::Document document;
            document.SetObject();
            rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

            rapidjson::Value document_obj(rapidjson::kObjectType);

            document_obj.AddMember("camera_id",
                    rapidjson::Value().SetString(m_camera_id.c_str(), 
                        allocator).Move(),allocator);
    //                 rapidjson::StringRef(m_camera_id.c_str()), allocator);
            document_obj.AddMember("timestamp", 
                    rapidjson::Value().SetString(record->timestamp.c_str(), 
                        allocator).Move(),allocator);
                
                object2json(record->object_list[i], document_obj, allocator, 
                        record->timestamp, image);
        
            document.AddMember("feature_data", document_obj, allocator);

            std::string r_string;
            rapidjson::StringBuffer stringbuf; 
            rapidjson::Writer<rapidjson::StringBuffer> writer(stringbuf);
//             rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(stringbuf);
            writer.SetMaxDecimalPlaces(6);
            document.Accept(writer);
            r_string = stringbuf.GetString();
            back_flag |= producekafka.push_data_to_kafka(r_string.c_str(), 
                    strlen(r_string.c_str()));
            LOG(INFO) << "record_json string_size:"<< r_string.size() 
                << "      size_of:"<< sizeof(r_string) 
                <<"       r_string:"<< r_string;
        }
    }
    return back_flag;
}
  
