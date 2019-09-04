#include "decode.h"


std::map<std::thread::id, int> thread_map;
std::vector<std::thread::id> thread_list;

int Decode::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}
int Decode::init(GrpcPram algorithm_source) {

	std::string image_path = "/data/wangxy/kafka_image";
	std::string brokers = "10.0.20.243:9092";

	int kafka_partition = -1;
    std::string topic = algorithm_source.uuid;
    std::vector<cv::Point> line = algorithm_source.line;
    int lines_interval = algorithm_source.lines_interval;
    
    m_gpu_id = algorithm_source.gpu_id;
    m_camera_id = algorithm_source.uuid;
    m_line = algorithm_source.line;
    m_lines_interval = algorithm_source.lines_interval;
    LOG_IF(INFO, 1) << "gpu_id:" << m_gpu_id;

	if (algorithm_source.source_type > 0)	{
		m_source_path = algorithm_source.source_name;
		m_source_type = algorithm_source.source_type;
	}
    LOG(INFO)<< "m_source_path - 3:" << m_source_path; 
	if (m_source_path.empty()){
        return -2;
	}
    algorithm.init(m_gpu_id, m_camera_id, image_path, kafka_partition, brokers, topic,
            line, lines_interval);

    std::stringstream image_str;
    image_str << image_path << "/" << m_camera_id << "/image_path/";
    m_image_path = image_str.str();
	std::string sql = "mkdir -p "+m_image_path+";";
    system( sql.c_str() );
}

int Decode::nv_decodeInit(GrpcPram &argv)
{

    const int gpuID = argv.gpu_id;

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (gpuID < 0 || gpuID >= nGpu)
    {
        std::ostringstream err;
        err << "Error: GPU ordinal out of range. Should be within [" << 0 << ", " 
            << nGpu - 1 << "]" << std::endl;
        throw std::invalid_argument(err.str());
    }
    CUdevice cuDevice;
    ck(cuDeviceGet(&cuDevice, gpuID));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    ck(cuCtxCreate(&g_cuContext, 0, cuDevice));

    init(argv);
}

int Decode::nv_encodeInit(RectSize &rectsize, DataBuffer &databuffer) {
    int quality = 64;
    // RectSize rectsize;
    // rectsize.height = width;
    // rectsize.width = height;
    // DataBuffer databuffer;
    databuffer.size= rectsize.height*rectsize.width*3 / 2;
    databuffer.capacity=databuffer.size;
    databuffer.mem_type = MEM_HOST;
    // CudaJpegEncode g_cudajpegencode;
    m_g_cudajpegencode.Init(quality, &rectsize, databuffer.mem_type);
    // cudaMallocManaged<unsigned char>(&img.data, img.width * img.height * img.channels * sizeof(unsigned char));
    databuffer.data = (char*)malloc(databuffer.capacity);//这里和下面num开辟的内存大小是一样的
    // cudaMalloc<char>(&databuffer.data, nWidth*nHeight*3 / 2);
}
/* 启动解码 */
int Decode::nv_decode(GrpcPram argv, int type)
{  
    // decode init
    nv_decodeInit(argv);
    if(m_source_type == 3) {
        Record record;
        std::map<std::string, Module_time> time_map;
        set_time_map(&time_map);
        algorithm.set_time_map(&time_map);

        record.timestamp = get_loacal_time(40);
		record.camera_id = m_camera_id;
        record.line = m_line;
        record.lines_interval = m_lines_interval;
        cv::Mat temp_image1 = cv::imread(m_source_path);
        cv::Mat temp_image;
        cv::cvtColor(temp_image1, temp_image, CV_BGR2BGRA);
        record.gpu_image.width = temp_image.cols;
        record.gpu_image.height = temp_image.rows;
        record.gpu_image.channels = 4;
        uint8_t *temp_gpu_data=NULL;
//         unsigned char *temp_gpu_data;
        std::cout <<"size:"<<4 * temp_image.cols * temp_image.rows<<std::endl;
        cudaMalloc((void **)&temp_gpu_data, 4 * temp_image.cols * temp_image.rows);
        std::cout<<"gpu_addr:"<<(void*)temp_gpu_data<<std::endl;
        cudaMemcpy(temp_gpu_data, temp_image.data, 4 * temp_image.cols * temp_image.rows, cudaMemcpyHostToDevice);
        record.gpu_image.data = temp_gpu_data;

        std::string image_name = m_image_path+record.timestamp+".jpg";
        imwrite(image_name, temp_image1);
        algorithm.inference(&record, type, image_name);
        return 0;
    }
    demuxer = new FFmpegDemuxer(m_source_path.c_str());
    dec = new NvDecoder(g_cuContext, demuxer->GetWidth(), demuxer->GetHeight(), 
            true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));
    int temp = 0;
    int nWidth = demuxer->GetWidth(), nHeight = demuxer->GetHeight();
    int nFrameSize = nWidth * nHeight * eOutputFormat;
    // std::unique_ptr<uint8_t[]> pImage(new uint8_t[nFrameSize]);
    img.channels = eOutputFormat;
    img.height = nHeight;
    img.width = nWidth;
    cudaMallocManaged<unsigned char>(&img.data, 
            img.width * img.height * img.channels * sizeof(unsigned char));

    // encode init
    RectSize rectsize;
    rectsize.height = img.height;
    rectsize.width = img.width;
    DataBuffer databuffer;
    nv_encodeInit(rectsize, databuffer);
    
    std::string out_file_path = "1.txt";
    std::ofstream out_file(out_file_path);
    if(!out_file.is_open()) {
        std::cout << "time_map2txt open file:" << out_file_path << " failed!" << std::endl;
        return -1;
    }
    do
    {
        demuxer->Demux(&pVideo, &nVideoBytes);
        std::cout << "nVideoBytes:" << nVideoBytes <<std::endl;
        dec->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        std::cout << "nFrameReturned:" << nFrameReturned <<std::endl;
        for (int i = 0; i < nFrameReturned; i++)
        {
            if (dec->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444){
                std::cout<<"Do YUV444ToColor32\n";
                YUV444ToColor32<BGRA32>((uint8_t*) ppFrame[i], dec->GetWidth(), 
                        img.data, eOutputFormat * dec->GetWidth(), 
                        dec->GetWidth(), dec->GetHeight());
            }
            else{
                Record record;
                std::map<std::string, Module_time> time_map;
                set_time_map(&time_map);
                algorithm.set_time_map(&time_map);
                std::chrono::high_resolution_clock::time_point time_start;
                std::chrono::high_resolution_clock::time_point time_end;
                std::chrono::milliseconds::duration::rep time_rep;

                time_start = std::chrono::high_resolution_clock::now(); 
                if(m_source_type == 1) {
                    record.timestamp = get_loacal_time(-1);
                }else if(m_source_type == 2) {
                    record.timestamp = get_loacal_time(40);
                }

                // gpu imwrite
                int num = dec->GetFrameSize() * sizeof(char);
                std::string image_name = m_image_path+record.timestamp+".jpg";
                cudaMemcpy(databuffer.data, ppFrame[i], num, cudaMemcpyDeviceToHost);
                // databuffer.data = (char*)ppFrame[i];
                m_g_cudajpegencode.SetData(&databuffer,&rectsize);
                m_g_cudajpegencode.EncodeJpeg(IMAGE_DEST_FILE, (char*)image_name.c_str(), databuffer.mem_type);


                std::cout<<"Do Nv12ToColor32\n";
                Nv12ToColor32<BGRA32>((uint8_t*) ppFrame[i], dec->GetWidth(), 
                        img.data, eOutputFormat * dec->GetWidth(), 
                        dec->GetWidth(), dec->GetHeight());
                record.gpu_image.data = img.data;
                record.gpu_image.height = img.height;
                record.gpu_image.width = img.width;
                record.gpu_image.channels = img.channels;
                if(!img.data){
                    LOG_IF(INFO, 1)<<"Image is empty!";
                    continue;
                }

                record.camera_id = m_camera_id;
                record.line = m_line;
                
                LOG_IF(INFO, 1) << "timestamp: " << record.timestamp;
                time_end = std::chrono::high_resolution_clock::now(); 
                time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
                (*m_time_map)["nv_decode"].total = time_rep;
                LOG_IF(INFO, 1) << "time of nv_decode inference:" << time_rep;
                time_start = std::chrono::high_resolution_clock::now();
                algorithm.inference(&record, type, image_name);
        //         sleep(1);
        //         cv::waitKey(40);
                time_end = std::chrono::high_resolution_clock::now(); 
                time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
                (*m_time_map)["inference"].total = time_rep;
                LOG_IF(INFO, 1) << "time of nv_decode total:" << time_rep;
                // time_map2txt(m_time_map, out_file);
            }
        }
        nFrame += nFrameReturned;
//     } while (nFrame<10);//while (nVideoBytes);
    } while (nVideoBytes);
    std::cout << "Total Frame decoded:" << nFrame << " " <<m_source_path << std::endl;
    // cuMemFree((CUdeviceptr)(img.data));
    delete demuxer;
    demuxer = nullptr;

    delete dec;
    dec = nullptr;
    thread_map[m_thread_id] = 0;
    out_file.close();
    return 0;
}

int Decode::cv_decodeInit(GrpcPram &argv) {

    LOG(INFO) << "argv.source_name:" << argv.source_name;
    init(argv);
}

int Decode::cv_decode(GrpcPram argv, int type) {
    LOG(INFO) << "before cvdecode init";
    cv_decodeInit(argv);
    
    LOG(INFO) << "after cvdecode init";
    if(m_source_type == 3) {
        Record record;
        std::map<std::string, Module_time> time_map;
        set_time_map(&time_map);
        algorithm.set_time_map(&time_map);
        record.timestamp = get_loacal_time(40);
        record.line = m_line;
        record.lines_interval = m_lines_interval;
        record.image = cv::imread(m_source_path);
		record.camera_id = m_camera_id;
		
        std::string image_name = m_image_path+record.timestamp+".jpg";
        imwrite(image_name, record.image);
        algorithm.inference(&record, type, image_name);
        return 0;
    }

    cv::VideoCapture capture;
    capture.open(m_source_path);
    if (!capture.isOpened()){
        LOG(ERROR)<<"Video:" << m_source_path <<" is not open!";
        return -1;
    }
    

    while(true){
        Record record;
        std::map<std::string, Module_time> time_map;
        set_time_map(&time_map);
        algorithm.set_time_map(&time_map);
        std::chrono::high_resolution_clock::time_point time_start;
        std::chrono::high_resolution_clock::time_point time_end;
        std::chrono::milliseconds::duration::rep time_rep;
        
        time_start = std::chrono::high_resolution_clock::now();
        cv::Mat image;
        capture>>image;
        if(image.empty()){
            LOG_IF(INFO, 1)<<"Image is empty!";
            capture.release();
            break;
        }
        if(m_source_type == 1) {
            LOG(INFO) << "get_loacal_time -1";
            record.timestamp = get_loacal_time(-1);
        }else if(m_source_type == 2) {
            LOG(INFO) << "get_loacal_time 40";
            record.timestamp = get_loacal_time(40);
        }

        std::string image_name = m_image_path+record.timestamp+".jpg";
        imwrite(image_name, image);

        record.image = image.clone();
		record.camera_id = m_camera_id;
        record.line = m_line;
		
        LOG_IF(INFO, 1) << "timestamp: " << record.timestamp;
        algorithm.inference(&record, type, image_name);
//         sleep(1);
//         cv::waitKey(40);
        time_end = std::chrono::high_resolution_clock::now(); 
        time_rep = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count(); 
        LOG_IF(INFO, 1) << "time of cv_decode:" << time_rep;
    }
    thread_map[m_thread_id] = 0;
    std::cout << "Total Frame decoded:" <<m_source_path << std::endl;
    return 0;
}


int Decode::start_thread(GrpcPram &argv, int type) {

    LOG(INFO) << "decode brefore start thread";
    std::thread* res;
    if(type == 1) {
        res = new std::thread(&Decode::cv_decode, this, argv, type);
    }else if(type == 2) {
        res = new std::thread(&Decode::nv_decode, this, argv, type);
    }
    m_thread_id = res->get_id();
    thread_list.push_back(m_thread_id);
    thread_map[m_thread_id] = 1;
    // sleep(10);
    LOG(INFO) << "decode after start thread";

}
