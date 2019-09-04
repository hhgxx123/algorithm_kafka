#ifndef __GLOG_INIT_H__ 
#define __GLOG_INIT_H__ 

#include <glog/logging.h>
#include <glog/raw_logging.h> //将信息输出到单独的文件和 LOG(ERROR) 
#include <stdlib.h>

//配置输出日志的文件夹：
#define LOGDIR "log"
#define MKDIR "mkdir -p " LOGDIR



class GLogHelper{ 
    public: //GLOG配置： 
    GLogHelper(char* program); //GLOG内存清理： 
    ~GLogHelper(); 
}; 


#endif
