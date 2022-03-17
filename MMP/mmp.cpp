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

#include "mmp.hpp"
#include "mmp_fn.hpp"

#define REGISTRATION strdup("/tmp/mmp")
#define MMP2SCH strdup("/tmp/mmp2sch")
#define SCH2MMP strdup("/tmp/sch2mmp")

int mmp2sch_fd = -1;
int sch2mmp_fd = -1;

int main(int argc, char **argv){

    _proc_list *proc_list = create_proc_list();
    int reg_fd = open_channel(REGISTRATION, O_RDONLY | O_NONBLOCK);
    
    // MMP 2 Scheduler 
    mmp2sch_fd = open_channel(MMP2SCH, O_WRONLY);
    sch2mmp_fd = open_channel(SCH2MMP, O_RDONLY | O_NONBLOCK);

    int fd_head;
    fd_set readfds;

    do{
        fd_head = make_fdset(&readfds, reg_fd, proc_list);
        
        if(select(fd_head + 1, &readfds, NULL, NULL, NULL)){
            if(FD_ISSET(reg_fd, &readfds)){
                check_registration(proc_list, reg_fd);
            }
            for(_proc *proc = proc_list->head; proc !=NULL; proc = proc->next){
                if(FD_ISSET(proc->request_fd, &readfds)){
                    request_handler(proc_list, proc);
                }
            }
            if(FD_ISSET(sch2mmp_fd, &readfds)){
                swapin(proc_list);
            }
        }
    }while(1);

    
    return 0;
}   
