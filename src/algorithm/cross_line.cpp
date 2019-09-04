#include "cross_line.h"

int M_CrossLineDetect::set_time_map(std::map<std::string, Module_time> *time_map) {
    m_time_map = time_map;
}

int M_CrossLineDetect::init(std::string &model_dir, CrossLineParas &crossline_pars, 
                const int gpu_id) {
    std::cout<<"CrossLineDetect init....."<<std::endl;
    m_crossline_detect.init(model_dir, crossline_pars, gpu_id);
    return 0;
}
int M_CrossLineDetect::cpu_pull_data(Record* record, CrossLineInputArray &obj_lists) {
    obj_lists.clear();
    for(int i=0; i<1; i++) {
        CrossLineInput obj_list;
        for(int j=0; j<record->object_list.size(); j++) {
            CrossLineObject cross_line_obj;
            cross_line_obj.object_id = record->object_list[j].object_id;
            cross_line_obj.region = record->object_list[j].region;
            obj_list.cross_object_list.push_back(cross_line_obj); 
        }
        LOG(INFO) << record->line;
        obj_list.line.x1 = record->line[0].x;
        obj_list.line.y1 = record->line[0].y;
        obj_list.line.x2 = record->line[1].x;
        obj_list.line.y2 = record->line[1].y;
        obj_list.lines_interval = record->lines_interval;
        obj_lists.push_back(obj_list);
    }
    if (obj_lists.size() > 0) {
        return 0;
    }else {
        return -1;
    }
}
int M_CrossLineDetect::inference(CrossLineInputArray &obj_lists, 
        CrossLineOutputArray &output_rets) {
    m_crossline_detect.inference(obj_lists, output_rets);
    return 0;
}
int M_CrossLineDetect::push_data(Record *record, CrossLineOutputArray &output_rets) {
    for(int i=0; i<1; i++) {
        CrossLineOutput output_ret = output_rets[i];
        LOG_IF(ERROR, record->object_list.size() != output_ret.cross_out_list.size()) 
            << "record->object_list.size():" << record->object_list.size()
            << "         output_ret.cross_out_list.size():" << output_ret.cross_out_list.size();
        assert(record->object_list.size() == output_ret.cross_out_list.size());

        record->come_num = output_ret.come_num;
        record->go_num = output_ret.go_num;
        record->line[0].x = output_ret.line.x1;
        record->line[0].y = output_ret.line.y1;
        record->line[1].x = output_ret.line.x2;
        record->line[1].y = output_ret.line.y2;
        record->lines_interval = output_ret.lines_interval;
        for(int i=0; i<output_ret.cross_out_list.size(); i++){
            record->object_list[i].touch_line_flag = output_ret.cross_out_list[i].touch_flag;
            record->object_list[i].first_touch_line = output_ret.cross_out_list[i].first_touch;
        }
    }
    return 0;
}
int M_CrossLineDetect::run(Record *record, int type) {
    CrossLineInputArray obj_lists;
    CrossLineOutputArray output_rets;
    if(cpu_pull_data(record, obj_lists) == 0) {
        inference(obj_lists, output_rets);
        push_data(record, output_rets);
    }
}
