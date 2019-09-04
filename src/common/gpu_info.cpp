#include "gpu_info.h"

GpuInfo::GpuInfo() {
    gpu_num = 0;
    retCode = RETURN_SUCCESS;

    // open the libnvidia-ml.so
    nvml_lib = NULL;
    nvml_lib = dlopen("libnvidia-ml.so", RTLD_NOW);
     if(nvml_lib == NULL){
        return;
    }

    CHECK_LOAD_NVML_FUNC(NVMLINIT, nvml_init, "nvmlInit");
    CHECK_LOAD_NVML_FUNC(NVMLSHUTDOWN, nvml_shutdown, "nvmlShutdown");

    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETCOUNT, 
            nvml_device_get_count, "nvmlDeviceGetCount");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETHANDLEBYINDEX, 
            nvml_device_get_handle_by_index, "nvmlDeviceGetHandleByIndex");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETMEMORYINFO, 
            nvml_device_get_memory_info, "nvmlDeviceGetMemoryInfo");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETUTILIZATIONRATES,
            nvml_device_get_utilization_rates, "nvmlDeviceGetUtilizationRates");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETTEMPERATURE, 
            nvml_device_get_temperature, "nvmlDeviceGetTemperature");
    check_nvml_errors(nvml_init());
    check_nvml_errors(nvml_device_get_count(&gpu_num));//【2】获得GPU设备的数量
    
    if(0) {
        gpu_fail:
            release();
    }
}

GpuInfo::~GpuInfo() {
    nvml_shutdown();
    //关闭动态库
    dlclose(nvml_lib);
}

int GpuInfo::get_gpu_num() {
    return gpu_num;
}


void GpuInfo::print_gpu_info(cudaDeviceProp strProp) {
    //【1】NIVIDA显卡的型号           
    LOG(INFO) << "gpu_info-- The name of Device: " << strProp.name;           
    //【2】设备全局内存的总量,单位:字节  
    LOG(INFO) << "gpu_info-- The totalGlobalMem of GPU: " << strProp.totalGlobalMem;
    //【3】在一个线程块Block中可以使用的最大共享内存的数量  
    LOG(INFO) << "gpu_info-- The sharedMemPerBlock of GPU: " 
        << strProp.sharedMemPerBlock; 
    //【4】每个线程块中可用的32位寄存器的数量  
    LOG(INFO) << "gpu_info-- The regsPerBlock of GPU: " << strProp.regsPerBlock;
    //【5】每一个线程束包含的线程的数量  
    LOG(INFO) << "gpu_info-- The warpSize: " << strProp.warpSize;
    //【6】内存复制中,最大的修正量  
    LOG(INFO) << "gpu_info-- The memPitch: " << strProp.memPitch;
    //【7】在一个线程块中,可以包含的最大线程数量  
    LOG(INFO) << "gpu_info-- The maxThreadPerBlock: " << strProp.maxThreadsPerBlock;
    //【8】常量内存的总量  
    LOG(INFO) << "gpu_info-- The totalConstMem: " << strProp.totalConstMem;
    LOG(INFO) << "gpu_info-- The major: " << strProp.major;  
    LOG(INFO) << "gpu_info-- The minor: " << strProp.minor;  
    LOG(INFO) << "gpu_info-- The multiProcessCount: " << strProp.multiProcessorCount;  
}

int GpuInfo::init() {
    std::printf("The number of GPU = %d\n", gpu_num);  
    for(int i=0; i<gpu_num; i++)        //【3】迭代的获取每一个【GPU设备】的属性  
    {  
        //【4】获取【GPU设备属性】的函数,并将获得的设备属性存放在strProp中  
        cudaGetDeviceProperties(&strProp,i);
        LOG(INFO) << "----General Information for device = " << i;  
        print_gpu_info(strProp);
        
        get_gpu_mem_data(i);
    }
}

void GpuInfo::get_gpu_mem_data(int gpu_index) {

    check_nvml_errors(nvml_device_get_handle_by_index(gpu_index, &device_handel));
    check_nvml_errors(nvml_device_get_memory_info(device_handel, &memory_info));
    check_nvml_errors(nvml_device_get_utilization_rates(device_handel, &gpu_utilization));
    LOG(INFO) <<"GPU: " << gpu_index << "   "
        << "Utilization:[gpu:" << gpu_utilization.gpu << " ," 
        << "memory:" << gpu_utilization.memory << "]   "
        << "Memory:[total:" << memory_info.total/1024/1024 << "MiB ,"
        << "free:" << memory_info.free/1024/1024 << "MiB ,"
        << "used:" << memory_info.used/1024/1024 << "MiB] ,  "
        << "free_rate:" << (float)memory_info.free/memory_info.total*100 << "%";
    if(0) {
        gpu_fail:
            release();
            LOG(ERROR) << "gpuinfo: get_gpu_mem_data error!";
    } 
}

float GpuInfo::get_gpu_freemem_rate(int gpu_index) {
    float free_mem_rete = 0;
    check_nvml_errors(nvml_device_get_handle_by_index(gpu_index, &device_handel));
    check_nvml_errors(nvml_device_get_memory_info(device_handel, &memory_info));
    
    free_mem_rete = (float)memory_info.free/memory_info.total*100;
    LOG(INFO) <<"GPU:" << gpu_index << "  free mem rate:" << free_mem_rete;
    
    if(0) {
        gpu_fail:
            release();
            LOG(ERROR) << "gpuinfo: get_gpu_mem_data error!";
    } 
    return free_mem_rete;
}
int GpuInfo::allot_gpu_index(int min_gpu_freemem_rate) {
    static int current_gpu_index = 0;
    static int no_freemem_count = 0;
    int back_value = 0;
    if(get_gpu_freemem_rate(current_gpu_index) > min_gpu_freemem_rate) {
        back_value = current_gpu_index;
        if(gpu_allot.count(current_gpu_index) > 0) {
            gpu_allot[current_gpu_index] += 1;
        }
        else {
            gpu_allot[current_gpu_index] = 1;
        }
        no_freemem_count = 0;
    }else {
        back_value = -1;
        no_freemem_count++;
    }
    
    current_gpu_index++;
    if(current_gpu_index >= gpu_num)
        current_gpu_index = 0;

    if(no_freemem_count >= gpu_num) {
        LOG(ERROR) << "allot_gpu_index: no free gpu!";
        return -2;
    }
    return back_value;
}

int GpuInfo::get_gpu_allot(int gpu_index) {
    int back_value = 0;
    if(gpu_allot.count(gpu_index) > 0) {
        back_value = gpu_allot[gpu_index];
    }

    return back_value;
}

void GpuInfo::release() {
    nvml_shutdown();
    //关闭动态库
    dlclose(nvml_lib);
}
