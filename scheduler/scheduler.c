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
FILE **fps;

int main(int argc, char **argv){
    int sync = find_int_arg(argc, argv, "-sync", 0);
    char * logdir = find_char_arg(argc, argv, "-logdir", ".");
    int init_sync = sync;
    set_priority(50); 
    set_affinity(0);
    

    /* LOG */
    fps = (FILE **)malloc(sizeof(FILE *)*sync);
    char logname[300];
    for(int i = 0; i < sync; i++){
        snprintf(logname,100,"%s/scheduler/sch_%d.log",logdir, i+1);
        fps[i] = fopen(logname, "a");
    }

    task_list_t *task_list = create_task_list();
    resource_t *gpu, *init_que;
    
    gpu = create_resource();
    init_que = create_resource();
        
    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);

#ifdef MMP
    mmp2sch_fd = open(MMP2SCH, O_RDONLY);
    sch2mmp_fd = open(SCH2MMP, O_WRONLY);
#endif

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
                    request_handler(task_list, task, gpu, init_que, current_time);

            if(!(init_que->waiting->count < init_sync)){
                init_sync = 0;
                if(init_que -> state == IDLE) target_pid = dequeue_backward(init_que->waiting, current_time, init_que);
                if(target_pid != -1) init_decision_handler(target_pid, task_list);
            }

            if( !(gpu->waiting->count < sync) && (init_que->waiting->count == 0)){
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
