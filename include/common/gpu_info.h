#ifndef __GPU_INFO_H__
#define __GPU_INFO_H__

#include <cuda_runtime.h>    //【1】CUDA运行时头文件,包含了许多的runtime API  
#include <device_launch_parameters.h>  
#include <driver_types.h>    //【2】驱动类型的头文件,包含cudaDeviceProp【设备属性】  
#include <cuda_runtime_api.h>//【3】cuda运行时API的头文件  
#include <stdio.h>  
#include <iostream>
#include <map>

#include "glog_init.h"



#include <string.h>
#include <sstream>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <dlfcn.h>
#define CUDAAPI
#define LOAD_FUNC(l, s) dlsym(l, s)
#define DL_CLOSE_FUNC(l) dlclose(l)


#define MIN_GPU_FREEMEM_RATE 50


typedef enum nvmlReturn_enum
{
    //!< The operation was successful
    NVML_SUCCESS = 0,
    //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_UNINITIALIZED = 1,
    //!< A supplied argument is invalid
    NVML_ERROR_INVALID_ARGUMENT = 2,
    //!< The requested operation is not available on target device
    NVML_ERROR_NOT_SUPPORTED = 3,
    //!< The current user does not have permission for operation
    NVML_ERROR_NO_PERMISSION = 4,
    //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    //!< A query to find an object was unsuccessful
    NVML_ERROR_NOT_FOUND = 6,
    //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    //!< A device's external power cables are not properly attached
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    //!< NVIDIA driver is not loaded
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    //!< User provided timeout passed
    NVML_ERROR_TIMEOUT = 10,
    //!< An internal driver error occurred
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef void * nvmlDevice_t;
typedef struct nvmlMemory_st
{
    unsigned long long total;   //!< Total installed FB memory (in bytes)
    unsigned long long free;    //!< Unallocated FB memory (in bytes)
    unsigned long long used;    //!< Allocated FB memory (in bytes). 
                                //Note that the driver/GPU always sets 
                                //aside a small amount of memory for bookkeeping
} nvmlMemory_t;

typedef struct nvmlUtilization_st
{
    unsigned int gpu;       //!< Percent of time over the past second during 
                            //which one or more kernels was executing on the GPU
    unsigned int memory;    //!< Percent of time over the past second during which 
                            //global (device) memory was being read or written
} nvmlUtilization_t;


typedef nvmlReturn_t(CUDAAPI *NVMLINIT)(void);  // nvmlInit
typedef nvmlReturn_t(CUDAAPI *NVMLSHUTDOWN)(void);  // nvmlShutdown 
// nvmlDeviceGetCount
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETCOUNT)(unsigned int *deviceCount);
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETHANDLEBYINDEX)(unsigned int index, 
        nvmlDevice_t *device); // nvmlDeviceGetHandleByIndex
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETMEMORYINFO)(nvmlDevice_t device, 
        nvmlMemory_t *memory); // nvmlDeviceGetMemoryInfo
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETUTILIZATIONRATES)(nvmlDevice_t device,
        nvmlUtilization_t *utilization); // nvmlDeviceGetUtilizationRates
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETTEMPERATURE)(nvmlDevice_t device, 
        int sensorType, unsigned int *temp); // nvmlDeviceGetTemperature

#define GPU_MAX_SIZE    128


#define RETURN_SUCCESS     0
#define RETURN_ERROR_LOAD_LIB       (-1)
#define RETURN_ERROR_LOAD_FUNC      (-2)
#define RETURN_ERROR_LIB_FUNC       (-3)
#define RETURN_ERROR_NULL_POINTER   (-4)


#define CHECK_LOAD_NVML_FUNC(t, f, s) \
do { \
    (f) = (t)LOAD_FUNC(nvml_lib, s); \
    if (!(f)) { \
        printf("Failed loading %s from NVML library\n", s); \
        retCode = RETURN_ERROR_LOAD_FUNC; \
         goto gpu_fail;\
    } \
} while (0)

static int check_nvml_error(int err, const char *func)
{
    if (err != NVML_SUCCESS) {
        printf(" %s - failed with error code:%d\n", func, err);
        return 0;
    }
    return 1;
}

#define check_nvml_errors(f) \
do{ \
    if (!check_nvml_error(f, #f)) { \
        retCode = RETURN_ERROR_LIB_FUNC; \
        goto gpu_fail;\
    }\
}while(0)



class GpuInfo {
    private:
        cudaDeviceProp strProp;  //【1】定义一个【设备属性结构体】的【结构体变量】  
        int retCode;
        void* nvml_lib;
        NVMLINIT                    nvml_init;
        NVMLSHUTDOWN                nvml_shutdown;
        NVMLDEVICEGETCOUNT          nvml_device_get_count;
        NVMLDEVICEGETHANDLEBYINDEX  nvml_device_get_handle_by_index;
        NVMLDEVICEGETMEMORYINFO     nvml_device_get_memory_info;
        NVMLDEVICEGETUTILIZATIONRATES       nvml_device_get_utilization_rates;
        NVMLDEVICEGETTEMPERATURE    nvml_device_get_temperature;

        nvmlDevice_t device_handel;
        
        nvmlMemory_t memory_info;
        nvmlUtilization_t gpu_utilization;

    private:
        unsigned int gpu_num;  
        std::map<int, int> gpu_allot;

    public:
        GpuInfo();
        ~GpuInfo();
        void release();
        void print_gpu_info(cudaDeviceProp strProp);
        int get_gpu_num();
        void get_gpu_mem_data(int gpu_index);
        float get_gpu_freemem_rate(int gpu_index);
        int init();
        int allot_gpu_index(int min_gpu_freemem_rate);
        int get_gpu_allot(int gpu_index);
};

static GpuInfo gpu_info;

#endif
