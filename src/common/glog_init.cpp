#include "glog_init.h"


//GLOG配置：
GLogHelper::GLogHelper(char* program)
{
    system( MKDIR );
    google::InitGoogleLogging(program);
    
    //glog停止输出
//     google::ShutdownGoogleLogging();

    //设置级别高于 google::INFO的日志同一时候输出到屏幕
    google::SetStderrLogging(google::INFO);
    FLAGS_colorlogtostderr=true;    //设置输出到屏幕的日志显示对应颜色

    //设置 google::ERROR 级别的日志存储路径和文件名称前缀
    //google::SetLogDestination(google::ERROR,"log/error_");

    //设置 google::INFO 级别的日志存储路径和文件名称前缀
    google::SetLogDestination(google::INFO,LOGDIR"/INFO_");

	//设置 google::WARNING 级别的日志存储路径和文件名称前缀
    google::SetLogDestination(google::WARNING,LOGDIR"/WARNING_");

    //设置 google::ERROR 级别的日志存储路径和文件名称前缀
    google::SetLogDestination(google::ERROR,LOGDIR"/ERROR_"); 
    FLAGS_logbufsecs =0;        //缓冲日志输出，默觉得30秒。此处改为马上输出
    FLAGS_max_log_size =100;  //最大日志大小为 100MB
    FLAGS_stop_logging_if_full_disk = true;     //当磁盘被写满时，停止日志输出

 	//设置文件名称扩展。如平台？或其他须要区分的信息
    google::SetLogFilenameExtension("91_"); 
    google::InstallFailureSignalHandler();      //捕捉 core dumped

 
}
//GLOG内存清理：
GLogHelper::~GLogHelper()
{
    google::ShutdownGoogleLogging();
}
