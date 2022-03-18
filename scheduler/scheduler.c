#include <fcntl.h>
#include <sys/types.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <float.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "scheduler.h"
#include "scheduler_fn.h"

#define REGISTRATION strdup("/tmp/scheduler")
#define MMP2SCH strdup("/tmp/mmp2sch")
#define SCH2MMP strdup("/tmp/sch2mmp")

int mmp2sch_fd = -1;
int sch2mmp_fd = -1;

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char **argv){
    int sync = find_int_arg(argc, argv, "-sync", 0);
    set_priority(50); 
    set_affinity(0);
    
    task_list_t *task_list = create_task_list();
    resource_t *gpu;
    
    gpu = create_resource();
        
    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);

    mmp2sch_fd = open(MMP2SCH, O_RDONLY);
    sch2mmp_fd = open(SCH2MMP, O_WRONLY);
    
    double current_time;
    int target_pid;
    int fd_head;
    fd_set readfds;
    task_info_t *task;
    
    do{
        target_pid = -1;

        fd_head = make_fdset(&readfds, reg_fd, task_list);
    
        if(select(fd_head +1, &readfds, NULL, NULL, NULL)){
            current_time = what_time_is_it_now();
            if(FD_ISSET(reg_fd, &readfds)) {
                check_registration(task_list, reg_fd, gpu);
            }

            for(task = task_list ->head; task !=NULL; task = task -> next) 
                if(FD_ISSET(task->request_fd, &readfds))
                    request_handler(task_list, task, gpu, current_time);

            if(!(gpu->waiting->count < sync)){
                if(sync){
                    send_release_time(task_list,current_time);
                    sync = 0;
                }
                if(gpu -> state == IDLE) target_pid = dequeue(gpu->waiting, current_time, gpu);
                if(target_pid != -1) decision_handler(target_pid, task_list);
            }
        }
    }while(!(task_list -> count == 0)); 
}   
