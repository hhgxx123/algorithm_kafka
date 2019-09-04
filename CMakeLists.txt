cmake_minimum_required(VERSION 3.5)
                           
project(main_wxy)

if(CMAKE_COMPILER_IS_GNUCC)
    message("COMPILER IS GNUCC")    
    ADD_DEFINITIONS ( -std=c++11 )  
endif(CMAKE_COMPILER_IS_GNUCC)


IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -ggdb3")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

##选用cpu版本，则定义该变量
# set(DEFINE_USE_CPU_IMAGE "USE_CPU_IMAGE")
IF (DEFINE_USE_CPU_IMAGE STREQUAL USE_CPU_IMAGE)
    ADD_DEFINITIONS(-DUSE_CPU_IMAGE)
endif()


##需要GRPC接口的支持，打开这一行
set(DEFINE_CIF_GRPC "_GRPC")
IF (DEFINE_CIF_GRPC STREQUAL _GRPC)
    ADD_DEFINITIONS(-D_CIF_GRPC)
    include_directories(/home/panhm/usr/aiit/eye/include)
endif()
message(DEFINE_CIF_GRPC=${DEFINE_CIF_GRPC})

# 2. CUDA
set(CUDA_DIR "/usr/local/cuda")
find_package(CUDA)
SET(CUDA_NVCC_FLAGS --disable-warnings;-std=c++11;-O3;-gencode arch=compute_75,code=sm_75)
#CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED)
message(${CUDA_NVCC_FLAGS})

# 3. opencv
link_directories("/usr/local/lib")
include_directories("/usr/local/include/opencv4")
link_directories("")

# 4. 英伟达硬编码
link_directories(${CUDA_DIR}/lib64)

# 4. tensorRT
include_directories(/data/wuh/software/TensorRT-5.1.2.2/include)
link_directories(/data/wuh/software/TensorRT-5.1.2.2/lib)

# 5.algorithm_module
#link_directories(/data/wuh/project/temp/algorithm_module/build)
link_directories(/home/wangxy/test_code/algorithm_module/build)


# 6. rapidjson
include_directories(${CMAKE_SOURCE_DIR}/lib) #该目录下放置包括rapidjson等的第三方库
                                            #rapidjson头文件中很多.h文件并不必要，递归包含会出问题

# 5. 头文件
# 5.1. 定义函数，用于递归添加头文件
function(include_sub_directories_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})               # 当前路径是一个目录吗，是的话就加入到包含目录
        message("include dir: " ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*) # 获得当前目录下的所有文件，让如ALL_SUB列表中
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            include_sub_directories_recursively(${root_dir}/${sub}) # 对子目录递归调用，包含
        endif()
    endforeach()
endfunction()
# 5.2. 添加头文件
include_sub_directories_recursively(${CMAKE_SOURCE_DIR}/include) # 对子目录递归调用，包含
MESSAGE(STATUS "CMAK_SOURCE_DIR" ${CMAKE_SOURCE_DIR})

# 6. 添加源文件
FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cu  ${CMAKE_SOURCE_DIR}/src/*.cpp )
message(${SOURCE_FILES})

# 7. 添加链接库
LINK_LIBRARIES(opencv_ml opencv_objdetect opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_shape opencv_videoio opencv_video)
LINK_LIBRARIES(nvcaffe_parser nvinfer nvinfer_plugin nvparsers)
LINK_LIBRARIES(avcodec avformat swscale avutil)
LINK_LIBRARIES(cuda nvcuvid)
LINK_LIBRARIES(nppisu nppicom nppig nppc)
LINK_LIBRARIES(algorithm)
LINK_LIBRARIES(glog)


# 9. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，
#但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
#SET(ALL_SRCS ${DIR_SRCS} ${PROTO_SRCS} ${SOURCE_FILES} ${M_INCLUDE_FILES})
SET(ALL_SRCS ${SOURCE_FILES} ${M_INCLUDE_FILES})

# 10.add executable file，添加要编译的可执行文件
# ADD_EXECUTABLE(${PROJECT_NAME} ${ALL_SRCS})
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${ALL_SRCS})

# 11. 链接目标文件与库文件，添加可执行文件所需要的库，比如我们用到了libm.so（命名规则：lib+name+.so），就添加该库的名称
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${PROTOBUF_LIBRARIES})
target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudart.so)
target_link_libraries(${PROJECT_NAME} /usr/local/cuda-10.1/lib64/libcudnn.so)
target_link_libraries(${PROJECT_NAME} rdkafka)


#GRPC
IF (DEFINE_CIF_GRPC STREQUAL _GRPC)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/lib/libz.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libaddress_sorting.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libgpr.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libgrpc.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libgrpc++.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libprotobufv3.so)
    target_link_libraries(${PROJECT_NAME} /home/panhm/.usr/aiit/eye/lib/libeyeggrpc.so)
endif()
