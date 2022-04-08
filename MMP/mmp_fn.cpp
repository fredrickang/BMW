#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include "mmp.hpp"
#include "mmp_fn.hpp"

#define string(x) #x
//#define MEM_LIMIT 40000
#define MEM_LIMIT 10737418240 // 10GB
//#define MEM_LIMIT 32212254720 // 30GB

extern int mmp2sch_fd;
extern int sch2mmp_fd;

static unsigned long long mem_current = 0;

using namespace std;

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

int make_fdset(fd_set *readfds,int reg_fd, _proc_list *proc_list, int swapin_fd){
    // initialize fd_set;
    FD_ZERO(readfds);

    FD_SET(swapin_fd, readfds);
    // set register_fd
    FD_SET(reg_fd, readfds);
        
    // if there exist registered dnn, set
    if(proc_list -> count > 0){
        _proc *node = proc_list -> head;
        while(node != NULL){
            FD_SET(node -> request_fd, readfds);
            node = node -> next;
        }
        return proc_list -> head ->request_fd;
    }
    return reg_fd;
}

_proc_list *create_proc_list(){
    _proc_list *tmp = (_proc_list *)malloc(sizeof(_proc_list));
    tmp->count = 0;
    tmp->head = NULL;
    return tmp;
}

void check_registration(_proc_list *proc_list, int reg_fd){
    reg_msg * msg = (reg_msg *)malloc(sizeof(reg_msg));
    
    while(read(reg_fd, msg, sizeof(reg_msg))>0){
        if(msg -> reg_type == 1) registration(proc_list, msg);
        else de_registration(proc_list, msg);
    }
}

void de_registration(_proc_list* proc_list, reg_msg *msg){

    _proc * target = find_proc_by_pid(proc_list, msg -> pid);
    size_t used_memory_size = getmemorysize(*(target->m_entry));
    
    close_channels(msg->pid);
    de_register_proc(proc_list, target);
    
    mem_current -= used_memory_size;
    
    DEBUG_PRINT(GREEN"Freed memory useage : %f\n"RESET, (float) used_memory_size/giga::num);
    DEBUG_PRINT(GREEN"Current memory useage : %f\n"RESET, (float) mem_current/giga::num );
}

void de_register_proc(_proc_list* proc_list, _proc * target){
    int target_id = target->id;
    int target_pid = target->pid;
    _proc *tmp, *prev;
    if(proc_list->head == target){
        proc_list->head = target->next;
        free(target);
        proc_list->count--;
        DEBUG_PRINT(BLUE"De-registration: [id] %d [pid] %d\n"RESET, target_id, target_pid);
        return;
    }
    tmp = proc_list->head;
    while(tmp!=target){
        prev = tmp;
        tmp = tmp->next;
    }
    prev->next = target->next;
    free(target);
    proc_list->count--;
    DEBUG_PRINT(BLUE"De-registration: [id] %d [pid] %d\n"RESET, target_id, target_pid);
    return;
}


void DEBUG_PRINT_PROCS(_proc_list * proc_list){
    _proc * head = proc_list -> head;
    
    if (head == NULL) DEBUG_PRINT(BLUE"Nothing registered"RESET);
    DEBUG_PRINT(BLUE"Process list: ")
    while( head != NULL){
        fprintf(stderr, "{%d / %d} ", head->id, head->pid);
        head = head->next;
    }
    fprintf(stderr, "\n"RESET);
}

void registration(_proc_list* proc_list, reg_msg *msg){
    _proc *proc = (_proc *)malloc(sizeof(_proc));
    proc -> id = proc_list->count;
    proc -> pid = msg -> pid;
    proc -> m_entry = new map<int, size_t>();

    DEBUG_PRINT(BLUE"Registration: [id] %d [pid] %d\n"RESET, proc->id, proc->pid);

    char req_fd_name[30];
    char dec_fd_name[30];

    snprintf(req_fd_name, 30, "/tmp/request_%d",proc->pid);
    snprintf(dec_fd_name, 30, "/tmp/decision_%d",proc->pid);

    proc -> request_fd = open_channel(req_fd_name, O_RDONLY);
    proc -> decision_fd = open_channel(dec_fd_name, O_WRONLY);

    DEBUG_PRINT(BLUE"Reqeust/decision channel opened [id] %d [pid] %d\n"RESET, proc->id, proc->pid);

    proc -> next = NULL;

    if(proc_list->head != NULL){
        proc -> next = proc_list -> head;
    }
    proc_list -> head = proc;
    proc_list -> count++;
    
    DEBUG_PRINT_PROCS(proc_list);
}

cudaAPI request_handler(_proc_list * proc_list, _proc * proc){
    req_msg *msg = (req_msg *)malloc(sizeof(req_msg));
    
    commErrchk(read(proc->request_fd, msg, sizeof(int)*REQ_MSG_SIZE));

    DEBUG_PRINT(GREEN"[REQEUST %d/%d] Index: %3d API: %15s Size: %5d\n"RESET, proc->id, proc->pid, msg->entry_index ,getcudaAPIString(msg->type), msg->size);
    
    if(msg->type == _Done_){
        DEBUG_PRINT(GREEN"Swap-in/out Done\n"RESET);
        return _Done_;
    }

    if(msg->type == _cudaMalloc_){
        /* Memory overflow handling */
        if(mem_current + msg->size > MEM_LIMIT){
            _proc* victim = choose_victim(proc_list, proc);
            if(victim == NULL){
                DEBUG_PRINT(RED"[Error] Victim not exist\n"RESET);
                exit(-1);
            }
            swapout(proc_list, victim, msg->size);
        }
        /* Update entry */
        proc->m_entry->insert(make_pair(msg->entry_index,msg->size));

        /* Update memory status */
        mem_current += msg->size;        
    }

    if(msg->type == _cudaFree_){
        mem_current -= proc->m_entry->at(msg->entry_index);
        /* Update entry*/
        proc->m_entry->erase(msg->entry_index);

    }    
    /* memory handling done! go do what ever you requested */
    int ack = 1;
    commErrchk(write(proc->decision_fd, &ack, sizeof(int)));
    return msg->type;
}

//  victim selection policy
//  Current policy: Random except request one
_proc* choose_victim(_proc_list* proc_list, _proc* proc){
    _proc * victim;
    list<int> pid_list;
    if(proc_list->count == 1) return NULL;

    int i = 0;
    for(_proc* tmp = proc_list->head; tmp != NULL; tmp = tmp->next){
        if(tmp->id != proc->id) pid_list.push_back(tmp->pid);
    }

    int victim_pid = pid_list.front();
    victim = find_proc_by_pid(proc_list, victim_pid);
    return victim;
}

// Page eviction protocal 
// Current policy: Greedy from oldest
void swapout(_proc_list* proc_list, _proc* proc, size_t size){
    size_t evict_size = 0;
    list<int> evict_entry_list;

    // find evict pages
    auto iter  = proc->m_entry->begin();
    while(iter != proc->m_entry->end() && (mem_current + size - evict_size > MEM_LIMIT)){
        evict_entry_list.push_back(iter->first);
        evict_size += iter->second;
        ++iter;
    }
    if(evict_entry_list.size() == 0){
        DEBUG_PRINT(RED"Victim(%d) has no pages to swap out\n"RESET, proc->pid);
        exit(-1);
    }
    // evict protocal
    // 1. wake the victim process
    // 2. send evict list
    int evict_entry_front = evict_entry_list.front();
    int evict_entry_back = evict_entry_list.back();

    int ack;
    evict_msg * msg =(evict_msg *)malloc(sizeof(evict_msg));
    msg->start_idx = evict_entry_front;
    msg->end_idx = evict_entry_back;

    DEBUG_PRINT(GREEN "[SWAP OUT] Victim: %d, Size: %d, Index: %d to %d\n" RESET, proc->pid, size, evict_entry_front, evict_entry_back);

    commErrchk(write(proc->decision_fd, msg, sizeof(int)*EVI_MSG_SIZE));    
   
    kill(proc->pid, SIGUSR1);

    cudaAPI ret;
    do{
        ret = request_handler(proc_list, proc);
    }while(ret != _Done_);
}

void swapin(_proc_list * proc_list){
    sch_msg *msg = (sch_msg *)malloc(sizeof(sch_msg));
    commErrchk(read(sch2mmp_fd, msg, sizeof(int)*SCH_MSG_SIZE));
    
    int target_pid = msg->pid;

    _proc * proc = find_proc_by_pid(proc_list, target_pid);
    kill(target_pid, SIGUSR2);
    cudaAPI ret;
    do{
        ret = request_handler(proc_list, proc);
    }while(ret != _Done_);
    
    // send to scheduler 
    int ack;
    commErrchk(write(mmp2sch_fd, &ack, sizeof(int)));
}

unsigned long long int getmemorysize(map<int,size_t> entry){
    unsigned long long int total_size = 0;
    for(auto iter = entry.begin(); iter != entry.end(); iter++){
        total_size += iter->second;
    }
    return total_size;
}



int open_channel(char *pipe_name,int mode){
    int pipe_fd;
    
    if( access(pipe_name, F_OK) != -1)
        remove(pipe_name);

    if( mkfifo(pipe_name, 0666) == -1){
        DEBUG_PRINT(RED"[ERROR]Fail to make pipe\n"RESET );
        exit(-1);
    }
    if( (pipe_fd = open(pipe_name, mode)) < 0){
        DEBUG_PRINT(RED"[ERROR]Fail to open channel for %s\n"RESET , pipe_name);
        exit(-1);
    }
    DEBUG_PRINT(BLUE"Channel %s opened\n"RESET, pipe_name);
   
   return pipe_fd;
}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaFree_:
            return string(_cudaFree_);
        case _Done_:
            return string(_Done_);
    }
}


void close_channel(int pid, char * pipe_name){
    if ( unlink(pipe_name) == -1){
        DEBUG_PRINT(RED"[%d] Fail to close channel %s\n"RESET, pid, pipe_name);
        exit(-1);
    }
    DEBUG_PRINT(BLUE"[%d] Channel %s closed\n"RESET, pid, pipe_name);
}

void close_channels(int pid){
    char request_name[30];
    char decision_name[30];
    
    snprintf(request_name, 30, "/tmp/request_%d", pid);
    snprintf(decision_name, 30, "/tmp/decision_%d", pid);
    
    close_channel(pid, request_name);
    close_channel(pid, decision_name);
}

_proc *find_proc_by_pid(_proc_list *proc_list, int pid){
    _proc * node = proc_list -> head;
    while(node -> pid != pid){
        node = node -> next;
    }
    return node;
}   

