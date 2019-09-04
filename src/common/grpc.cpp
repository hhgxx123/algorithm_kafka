#include "grpc.h"

Grpc::Grpc() {
}

int Grpc::feature2json(std::string reid_name,std::string &img_file,
        std::vector<float> &feature,std::string &str_json){
    rapidjson::Document document;
    rapidjson::Value document_obj(rapidjson::kObjectType);
    document.SetObject();
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
    rapidjson::Value reid_json_array(rapidjson::kArrayType);

    for(int i=0;i<feature.size();i++){
        reid_json_array.PushBack(feature[i], allocator);
    }
    document_obj.AddMember("file",rapidjson::Value().SetString(img_file.c_str(), 
                allocator).Move(), allocator);
    if(!reid_json_array.Empty() > 0) {
        document_obj.AddMember("reid", reid_json_array, allocator);
    }
    document.AddMember(rapidjson::Value().SetString(reid_name.c_str(),
                allocator).Move(), document_obj, allocator);
    rapidjson::StringBuffer stringbuf;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(stringbuf);
    writer.SetMaxDecimalPlaces(6);
    document.Accept(writer);
    str_json = stringbuf.GetString();
    //LOG(INFO)<<reid_name+"_json="+str_json<<std::endl;
    return 0;
}


int Grpc::image_FaceReid(std::string &img_file,std::string &str_json,int gpu_id){
    std::vector<float> feature;

    cv::Mat image = cv::imread(img_file);
    if (image.empty())
        return -1;
    feature.clear();
    std::unique_lock <std::mutex> lk(faceReid_mutex);
    if(faceReid_Flag==0){
        algorithm.init_FaceReid(gpu_id);
        faceReid_Flag=1;
    }
    algorithm.inference_FaceReid(image,feature);
    //algorithm.release();
    feature2json("face_reid",img_file,feature,str_json);
    LOG(INFO)<<"json="+str_json<<std::endl;

    return 0;
}

int Grpc::image_PersonReid(std::string &img_file,std::string &str_json,int gpu_id){
    std::vector<float> feature;

    cv::Mat image = cv::imread(img_file);
    if (image.empty())
        return -1;

    feature.clear();
    std::unique_lock <std::mutex> lk(personReid_mutex);
    if(personReidFlag==0){
        algorithm.init_PersonReid(gpu_id);
        personReidFlag=1;
    }
    algorithm.inference_PersonReid(image,feature);
    //algorithm.release();

    feature2json("person_reid",img_file,feature,str_json);
    LOG(INFO)<<"json="+str_json<<std::endl;

    return 0;
}

int pipeLineImgFeature_fun(pipeLineSourceImage request,pipelineFeatureResponse *reply){
    int Ret;
    GpuInfo gpu_info;
    Grpc grpc;
    gpu_info.init();
    std::string  reid_json;
    std::string  img_file=request.file_name;
    int gpu_id = request.gpu_id;

    if(faceReid_Flag==0){
        if(gpu_id < 0) {
            while((gpu_id = gpu_info.allot_gpu_index(MIN_GPU_FREEMEM_RATE)) < 0)
                usleep(300000);
        }
    }
    if(personReidFlag==0){
        if(gpu_id < 0) {
            while((gpu_id = gpu_info.allot_gpu_index(MIN_GPU_FREEMEM_RATE)) < 0)
                usleep(300000);
        }
    }

    if (1==request.feature_type){
        Ret=grpc.image_FaceReid(request.file_name,reid_json,gpu_id);
    }else if(2==request.feature_type){
        Ret=grpc.image_PersonReid(request.file_name,reid_json,gpu_id);
    }

    if (Ret<0){
        reply->results=Ret;
    }else {
        reply->results = 0;
    }
    reply->file_name=request.file_name;
    reply->gpu_id=gpu_id;
    reply->feature=reid_json;


    return 0;
}


int pipelineaddSource_fun(pipelineSourceData grpc_source_data,pipelineResponse *reply) {

    int results;
    int gpu_id;
    GpuInfo gpu_info;
    gpu_info.init();
    int type; //type=1,cpu   2,gpu
#ifdef USE_CPU_IMAGE
    type = 1; //type=1,cpu   2,gpu
#else
    type = 2; //type=1,cpu   2,gpu
#endif   
    //创建线程
    for(int i=0; i<50; i++){
        if(i >= thread_list.size() || 
                (i < thread_list.size() && thread_map[thread_list[i]] == 0)) {
            GrpcPram souceData;

            souceData.source_type=grpc_source_data.source_type;//1,rtsp  2,video  3,picture
            souceData.source_name=grpc_source_data.source_name;
            souceData.uuid=grpc_source_data.uuid;

            gpu_id = grpc_source_data.gpu_id;
            if(gpu_id < 0) {
                while((gpu_id = gpu_info.allot_gpu_index(MIN_GPU_FREEMEM_RATE)) < 0);
                grpc_source_data.gpu_id=gpu_id;
            }
            souceData.gpu_id=gpu_id;

            souceData.valid=grpc_source_data.valid;
            souceData.line = {cv::Point(300, 600), cv::Point(1220, 600)};
            souceData.lines_interval=5;
        	Decode *decode = new Decode();

            decode->start_thread(souceData, type);
            results=0;
            break;
       }
    }
    if (NULL!=reply){
        reply->results=results;
        reply->gpu_id=gpu_id;
        reply->valid=1;
    }
    for(int i=0; i<gpu_info.get_gpu_num(); i++) {
        LOG(INFO) << "gpu_allot " << i << " : " << gpu_info.get_gpu_allot(i);
    }
    return 0;
}



#ifdef _CIF_GRPC
int Grpc_run(std::string &port) {
    __GRPC_CALLBACL_FUN  cifs_fun;
    cifs_fun.pipelineaddSource=pipelineaddSource_fun;
    cifs_fun.pipeLineImgFeature=pipeLineImgFeature_fun;
    grpc_RunServer((void *)(&cifs_fun),port);

}
#endif
