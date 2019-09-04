#include "kafka.h"
#include <errno.h>

/*
    每条消息调用一次该回调函数，说明消息是传递成功(rkmessage->err == RD_KAFKA_RESP_ERR_NO_ERROR)
    还是传递失败(rkmessage->err != RD_KAFKA_RESP_ERR_NO_ERROR)
    该回调函数由rd_kafka_poll()触发，在应用程序的线程上执行
 */
/*
static void dr_msg_cb(rd_kafka_t *rk,
                      const rd_kafka_message_t *rkmessage, void *opaque){
    if(rkmessage->err)
        fprintf(stderr, "%% Message delivery failed: %s\n",
                rd_kafka_err2str(rkmessage->err));
    else
        fprintf(stderr,
                "%% Message delivered (%zd bytes, "
                "partition %"PRId32")\n",
                rkmessage->len, rkmessage->partition);
        //rkmessage被librdkafka自动销毁
}
*/

int ProducerKafka::init(int partition, const char *brokers,const char *topic)
{
    char tmp[16]={0};  
    char errstr[512]={0};

    partition_ = partition;

    /* Kafka configuration */
    conf_ = rd_kafka_conf_new();

    //set logger :register log function
    rd_kafka_conf_set_log_cb(conf_, logger);

    /*topic configuration*/
    topic_conf_ = rd_kafka_topic_conf_new();

    if (!(handler_  = rd_kafka_new(RD_KAFKA_PRODUCER, conf_, errstr, sizeof(errstr))))
    {
        fprintf(stderr, "*****Failed to create new producer: %s*******\n",errstr);
        return PRODUCER_INIT_FAILED;
    }

    rd_kafka_set_log_level(handler_, LOG_DEBUG);

    /* Add brokers */
    if (rd_kafka_brokers_add(handler_, brokers) == 0)
    {
        fprintf(stderr, "****** No valid brokers specified********\n");
        return PRODUCER_INIT_FAILED;
    }

    /* Create topic */
    topic_ = rd_kafka_topic_new(handler_, topic, topic_conf_);

    return 0;
}  
  
void ProducerKafka::release()  
{  
    /* Destroy topic */  
    rd_kafka_topic_destroy(topic_);  
  
    /* Destroy the handle */  
    rd_kafka_destroy(handler_);  
}  
  
int ProducerKafka::push_data_to_kafka(const char* buffer, const int buf_len)  
{  
    int ret;  
    char errstr[512]={0};  
      
    if(NULL == buffer)  
        return 0;
    retry:
    ret = rd_kafka_produce(topic_, partition_, RD_KAFKA_MSG_F_COPY,
                            (void*)buffer, (size_t)buf_len, NULL, 0, NULL);  
  
    if(ret == -1)  
    {

        if (rd_kafka_errno2err(errno) == RD_KAFKA_RESP_ERR__QUEUE_FULL){
            /*如果内部队列满，等待消息传输完成并retry,
            内部队列表示要发送的消息和已发送或失败的消息，
            内部队列受限于queue.buffering.max.messages配置项*/
            rd_kafka_poll(handler_, 1000);
            goto retry;
        }

        fprintf(stderr,"****Failed to produce to topic %s partition %i: %s*****\n",  
            rd_kafka_topic_name(topic_), partition_,  
            rd_kafka_err2str(rd_kafka_errno2err(errno)));  
      
        rd_kafka_poll(handler_, 0);  
        return PUSH_DATA_FAILED;  
    }  
      
    fprintf(stderr, "***Sent %d bytes to topic:%s partition:%i*****\n",  
        buf_len, rd_kafka_topic_name(topic_), partition_);  
  
    rd_kafka_poll(handler_, 0);  
  
    return PUSH_DATA_SUCCESS;  
}  
  
int produce_data()  
{  
    char test_data[100];  
    strcpy(test_data, "helloworld");  
  
    ProducerKafka* producer = new ProducerKafka();  
    if (PRODUCER_INIT_SUCCESS == producer->init(RD_KAFKA_PARTITION_UA, 
                "10.0.23.131:9092,10.0.23.133:9092,10.0.23.134:9092", "test-topic"))  
    {  
        printf("producer init success\n");  
    }  
    else  
    {  
        printf("producer init failed\n");  
        return 0;  
    }  
      
    while (fgets(test_data, sizeof(test_data), stdin)) {  
        size_t len = strlen(test_data);  
        if (test_data[len - 1] == '\n')  
            test_data[--len] = '\0';  
        if (strcmp(test_data, "end") == 0)  
            break;  
        if (PUSH_DATA_SUCCESS == producer->push_data_to_kafka(test_data, strlen(test_data)))  
            printf("push data success %s\n", test_data);  
        else  
            printf("push data failed %s\n", test_data);  
    }  
  
    producer->release();  
      
    return 0;     
} 

int get_data()
{
    char test_data[100];  
    strcpy(test_data, "helloworld");  
  
    ProducerKafka* producer = new ProducerKafka();  
    if (PRODUCER_INIT_SUCCESS == producer->init(0, 
                "10.0.23.131:9092,10.0.23.132:9092,10.0.23.133:9092", "test-topic"))  
    {  
        printf("producer init success\n");  
    }  
    else  
    {  
        printf("producer init failed\n");  
        return 0;  
    }  
      
    while (fgets(test_data, sizeof(test_data), stdin)) {  
        size_t len = strlen(test_data);  
        if (test_data[len - 1] == '\n')  
            test_data[--len] = '\0';  
        if (strcmp(test_data, "end") == 0)  
            break;  
        if (PUSH_DATA_SUCCESS == producer->push_data_to_kafka(test_data, 
                    strlen(test_data)))  
            printf("push data success %s\n", test_data);  
        else  
            printf("push data failed %s\n", test_data);  
    }  
  
    producer->release();  
      
    return 0;     
}
/*
int main()
{
    produce_data();
}
*/
