//
// Created by panhm on 2019-08-05.
//

#ifndef GRPC_TEST_GRPC_EYE_SERVER_H
#define GRPC_TEST_GRPC_EYE_SERVER_H


#include <iostream>
#include <vector>
#include <string>

typedef struct {
    int source_type;
    std::string source_name;
    std::string uuid;
    int gpu_id;
    int valid;
    std::vector<cv::Point> line;
    int lines_interval;
    pthread_t algorithm_t;
}GrpcPram;

typedef struct {
    int source_type;
    std::string source_name;
    std::string uuid;
    int gpu_id;
    int valid;
}pipelineSourceData;


typedef struct{
    int results;
    int gpu_id;
    int valid;
}pipelineResponse;


typedef struct{
        std::string file_name;
        int  feature_type;
        int  gpu_id;
        std::string rule;
}pipeLineSourceImage;

typedef struct{
        std::string file_name;
        int         results;
        int         gpu_id;
        std::string feature;
}pipelineFeatureResponse;


typedef struct
{
    int (*pipelineaddSource)(pipelineSourceData   source_data,pipelineResponse *reply);
    int (*pipeLineImgFeature)(pipeLineSourceImage request,pipelineFeatureResponse *reply);
}__GRPC_CALLBACL_FUN;




extern void grpc_RunServer(void *args,std::string server_address);

#endif //GRPC_TEST_GRPC_EYE_SERVER_H
