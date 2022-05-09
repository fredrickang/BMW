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
#include "scheduler.hpp"
#include "scheduler_fn.hpp"


#define MEM_LIMIT 10737418240
#define string(x) #x

static size_t mem_current = 0;

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

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
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

void set_priority(int priority){
    struct sched_param prior;
    memset(&prior, 0, sizeof(prior));
    prior.sched_priority = priority;
    if(sched_setscheduler(getpid(), SCHED_FIFO, &prior) == -1) perror("SCHED_FIFO :");
}

void set_affinity(int core){
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

/* DNN List Functions */


#ifdef DEBUG
void print_list(char * name, task_list_t * task_list){
    task_info_t * head = task_list -> head;
    DEBUG_PRINT(BLUE"%s : ", name);
    if (head == NULL) fprintf(stderr, "Nothing registered");
    while( head != NULL){
        fprintf(stderr, "{[%d] %d} ", head->pid, head->priority);
        head = head->next;
    }
    fprintf(stderr,"\n"RESET);
}

void print_queue(char * name, queue_t * q){
    node_t * head =  q -> front;
    DEBUG_PRINT(BLUE"%s : ", name);
    if (head == NULL) 
        fprintf(stderr, "Empty Queue"); 
  
    while (head != NULL) { 
        fprintf(stderr, "{[%d %d]} ", head -> pid, head->priority);
        head = head->next; 
    }
    fprintf(stderr,"\n"RESET);
}
#else
void print_list(char *name, task_list_t * task_list){
}
void print_queue(char * name, queue_t *q){

}
#endif

task_list_t *create_task_list(){
    task_list_t *tmp = (task_list_t *)malloc(sizeof(task_list_t));
    tmp->count = 0;
    tmp->head = NULL;
    return tmp;
}

void register_task(task_list_t *task_list, task_info_t *task){
    DEBUG_PRINT(BLUE"Task(%d) Registration\n"RESET,task->pid);
    if(task_list->head == NULL){
        task_list->head = task;
        task_list->count++;
        print_list("TaskList", task_list);
        return;
    }
    task->next = task_list->head;
    task_list->head = task;
    task_list->count++;
    print_list("TaskList", task_list);
    return;
}

void de_register_task(task_list_t *task_list, task_info_t *task){ 
    DEBUG_PRINT(BLUE"Task(%d) De-registration\n"RESET,task->pid);
    task_info_t *tmp, *prev;    
    if(task_list->head == task){
        task_list->head = task->next;
        free(task);
        task_list->count--;
        print_list("TaskList", task_list);
        return;
    }
    tmp = task_list->head;
    while(tmp != task){
        prev = tmp;
        tmp = tmp->next;
    }
    prev->next = task->next;
    free(task);
    task_list->count--;
    print_list("TaskList", task_list);
    return;
} 



/* Resources */

resource_t *create_resource(){
    resource_t *tmp = (resource_t *)malloc(sizeof(resource_t));
    tmp -> state = IDLE;
    tmp -> pid = -1;
    tmp -> waiting = create_queue();
    return tmp;
}

/* Waiting Queue Things */

node_t* new_node(int pid,int priority){
    node_t *tmp = (node_t *)malloc(sizeof(node_t));
    tmp -> pid = pid;
    tmp -> priority = priority;
    tmp -> next = NULL;
    return tmp;
}

queue_t *create_queue(){
    queue_t *q = (queue_t *)malloc(sizeof(queue_t));
    q->front = NULL;
    q->count = 0;
    return q; 
}

int enqueue(queue_t *q, int pid, int priority){
    DEBUG_PRINT(BLUE"Enqueue Job(%d %d)\n"RESET,pid, priority);
    node_t *tmp = new_node(pid, priority);
    q->count ++;    
    
    if(q->front == NULL){
        q->front = tmp;
        print_queue("WQ", q);
        return -1;
    }    

    if(q->front -> priority > tmp-> priority ){
        tmp->next = q->front;
        q->front = tmp;
    }
    else{
        node_t *iter = q->front;
        while(iter->next != NULL && iter->next->priority < tmp->priority){
            iter = iter->next;
        }
        tmp->next = iter->next;
        iter->next = tmp;
    }
    print_queue("WQ", q);
}

int enqueue_backward(queue_t *q, int pid, int priority){
    DEBUG_PRINT(BLUE"Enqueue Init Job(%d %d)\n"RESET,pid, priority);
    node_t *tmp = new_node(pid, priority);
    q->count ++;    
    
    if(q->front == NULL){
        q->front = tmp;
        print_queue("WQ", q);
        return -1;
    }    

    if(q->front -> priority < tmp-> priority){
        tmp->next = q->front;
        q->front = tmp;
    }
    else{
        node_t *iter = q->front;
        while(iter->next != NULL && iter->next->priority < tmp->priority){
            iter = iter->next;
        }
        tmp->next = iter->next;
        iter->next = tmp;
    }
    print_queue("WQ", q);
}

int dequeue(queue_t *q, resource_t *res){
    if (q -> front == NULL){
        //DEBUG_PRINT(RED"Waiting Queue empty\n"RESET);
        return -1;
    }
    
    node_t *target = q->front;
    
    DEBUG_PRINT(BLUE"Dequeue Job(%d %d)\n"RESET,target->pid, target->priority);
    q->front = target->next;

    int target_pid = target->pid;

    res -> state = BUSY;
    res -> pid  = target_pid;  
    res -> scheduled = what_time_is_it_now();
    
    q -> count --;
    free(target);
    print_queue("WQ", q);
    return target_pid;
}  

int dequeue_backward(queue_t *q, resource_t *res){
    if (q -> front == NULL){
        //DEBUG_PRINT(RED"Waiting Queue empty\n"RESET);
        return -1;
    }
    
    node_t *target = q->front;
    node_t *prev = NULL;
    while(target->next != NULL){
        prev = target;
        target = target->next;
    }
    if(prev == NULL){
        q->front = NULL;
    } 
    else{
        prev->next = NULL;
    }
    DEBUG_PRINT(BLUE"Dequeue Job(%d %d)\n"RESET,target->pid, target->priority);
    
    int target_pid = target->pid;

    res -> state = BUSY;
    res -> pid  = target_pid;  
    res -> scheduled = what_time_is_it_now();
    
    q -> count --;
    free(target);
    print_queue("WQ", q);
    return target_pid;
}  

void send_release_time(task_list_t *task_list){
    struct timespec release_time;
    clock_gettime(CLOCK_MONOTONIC, &release_time);

    for(task_info_t * node = task_list -> head ; node != NULL ; node = node -> next){
        if(write(node->sch_dec_fd, &release_time,sizeof(struct timespec)) < 0)
            perror("decision_handler");  
    }
}

task_info_t *find_task_by_pid(task_list_t *task_list, int pid){
    task_info_t * node = task_list -> head;
    while(node -> pid != pid){
        node = node -> next;
    }
    return node;
}   

// Register //

void check_registration(task_list_t *task_list, int reg_fd, resource_t *res){
    reg_msg * msg = (reg_msg *)malloc(sizeof(reg_msg));
        
    while(read(reg_fd, msg, sizeof(reg_msg)) > 0){
        if(msg -> regist == 1) do_register(task_list, msg); 
        else deregister(task_list, msg, res);
    }
}

void do_register(task_list_t *task_list, reg_msg *msg){
    task_info_t *task = (task_info_t *)malloc(sizeof(task_info_t));
    task -> pid = msg -> pid;
    task -> id = task_list->count;
    task -> priority = msg -> priority;
    task -> m_entry = new map<int, size_t>();
    task -> scheduled_time = 0;

    DEBUG_PRINT(BLUE"======== REGISTRATION ========\n"RESET);
    DEBUG_PRINT(BLUE"[PID]      %3d\n"RESET, task-> pid);
    DEBUG_PRINT(BLUE"[Priority] %3d\n"RESET, task->priority);
    
    char sch_req_fd_name[50];
    char sch_dec_fd_name[50];
    char mm_req_fd_name[50];
    char mm_dec_fd_name[50];

    snprintf(sch_req_fd_name, 50,"/tmp/sch_request_%d",task->pid);
    snprintf(sch_dec_fd_name, 50,"/tmp/sch_decision_%d",task->pid);
    snprintf(mm_req_fd_name, 50,"/tmp/mm_request_%d",task->pid);
    snprintf(mm_dec_fd_name, 50,"/tmp/mm_decision_%d",task->pid);

    task -> sch_req_fd = open_channel(sch_req_fd_name, O_RDONLY);
    task -> sch_dec_fd = open_channel(sch_dec_fd_name, O_WRONLY);
    task -> mm_req_fd = open_channel(mm_req_fd_name, O_RDONLY);
    task -> mm_dec_fd = open_channel(mm_dec_fd_name, O_WRONLY);
    
    task -> next = NULL;

    register_task(task_list, task);    
}


void deregister(task_list_t *task_list, reg_msg *msg, resource_t *res){
    
    task_info_t *target = find_task_by_pid(task_list, msg -> pid);
    size_t used_memory_size = getmemorysize(*(target->m_entry));
    int pid=target -> pid;

    close_channels(target);
    de_register_task(task_list, target);
    
    mem_current -= used_memory_size;
    write(target->sch_dec_fd, &pid, sizeof(int));
    
    /* LOG */
#ifdef LOG
    fclose(fps[target->priority-1]);
#endif     

    if (res-> pid == pid) res->state = IDLE;
    DEBUG_PRINT(GREEN"Freed memory useage : %f\n"RESET, (float) used_memory_size/giga::num);
    DEBUG_PRINT(GREEN"Current memory useage : %f\n"RESET, (float) mem_current/giga::num );
}

// Request Handler //

void sch_request_handler(task_list_t *task_list, task_info_t *task, resource_t *res, resource_t *init_que){    
    int ack;
    commErrchk(read(task -> sch_req_fd, &ack, sizeof(int)));
    
    if(ack == 99){
        enqueue(init_que->waiting, task->pid, task->priority);
        return;
    }
    if(init_que -> state == BUSY && init_que -> pid == task -> pid){
        DEBUG_PRINT(GREEN"Init done(%d)\n"RESET,task->priority);
#ifdef LOG
        fprintf(fps[task->priority-1],"%f,",(what_time_is_it_now() - init_que->scheduled));
#endif
        init_que -> state = IDLE;
        init_que -> pid = -1;
    }
    
    if(res -> state == BUSY && res -> pid == task->pid){ /* Job termniation */
        DEBUG_PRINT(GREEN"Term Job(%d)\n"RESET,task->priority);
#ifdef LOG
        fprintf(fps[task->priority-1],"%f\n",(what_time_is_it_now() - res->scheduled));
#endif 
        res -> state = IDLE;
        res -> pid = -1;
    }
    else{ /* Job release */
        DEBUG_PRINT(GREEN"Release Job(%d)\n"RESET,task->priority);
        enqueue(res->waiting, task->pid, task->priority);
    }
}

cudaAPI mm_request_handler(task_list_t * proc_list, task_info_t * proc){
    req_msg *msg = (req_msg *)malloc(sizeof(req_msg));
    
    commErrchk(read(proc->mm_req_fd, msg, sizeof(req_msg)));

    DEBUG_PRINT(GREEN"[REQEUST %d/%d] Index: %3d API: %15s Size: %lu\n"RESET, proc->id, proc->pid, msg->entry_index ,getcudaAPIString(msg->type), msg->size);
    
    if(msg->type == _SWAPIN_){
        DEBUG_PRINT(GREEN "[SWAP IN] Reqested: %d, Size: %lu\n"RESET, proc->pid, msg->size);
        if(mem_current + msg->size > MEM_LIMIT){
            size_t should_swap_out = mem_current + msg->size - MEM_LIMIT;
            size_t swap_outed = 0;
            while(should_swap_out > swap_outed){
                task_info_t * victim = choose_victim(proc_list, proc);
                if(victim == NULL){
                    DEBUG_PRINT(RED"[Error] Victim not exist\n"RESET);
                    exit(-1);
                }
                swap_outed += swapout(proc_list, victim, (should_swap_out - swap_outed));
            }
        }
    }

    if(msg->type == _Done_){
        DEBUG_PRINT(GREEN"Swap-in/out Done\n"RESET);
        return _Done_;
    }

    if(msg->type == _cudaMalloc_){
        /* Memory overflow handling */
        if(mem_current + msg->size > MEM_LIMIT){
            size_t should_swap_out = mem_current + msg->size - MEM_LIMIT;
            size_t swap_outed = 0;
            while(should_swap_out > swap_outed){
                task_info_t * victim = choose_victim(proc_list, proc);
                if(victim == NULL){
                    DEBUG_PRINT(RED"[Error] Victim not exist\n"RESET);
                    exit(-1);
                }
                swap_outed += swapout(proc_list, victim, (should_swap_out - swap_outed));
            }
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
    commErrchk(write(proc->mm_dec_fd, &ack, sizeof(int)));
    return msg->type;
}

task_info_t* choose_victim(task_list_t* proc_list, task_info_t* proc){
    task_info_t * victim;
    if(proc_list->count == 1) return NULL;

    double latest_proc_scheduled_time = -1;
    int latest_proc_pid = -1;

    for(task_info_t* tmp = proc_list->head; tmp != NULL; tmp = tmp->next){
        if(tmp->scheduled_time >= latest_proc_scheduled_time && (getmemorysize(*(tmp->m_entry))!=0) && tmp != proc){
            latest_proc_scheduled_time = tmp->scheduled_time;
            latest_proc_pid = tmp->pid;
        }
    }
    
    if (latest_proc_pid == -1) return NULL;

    victim = find_task_by_pid(proc_list, latest_proc_pid);
    return victim;
}


size_t getmemorysize(map<int,size_t> entry){
    size_t total_size = 0;
    for(auto iter = entry.begin(); iter != entry.end(); iter++){
        total_size += iter->second;
    }
    return total_size;
}


void init_decision_handler(int target_pid, task_list_t *task_list){

    int ack = 0;
    task_info_t *target = find_task_by_pid(task_list, target_pid);

    DEBUG_PRINT(GREEN"Scheduled Job(%d)\n"RESET,target->priority);    
    commErrchk(write(target->sch_dec_fd,&ack,sizeof(int)));
}

void decision_handler(int target_pid, task_list_t *task_list){

    int ack = 0;
    task_info_t *target = find_task_by_pid(task_list, target_pid);

    //
    sch_msg * msg = (sch_msg *)malloc(sizeof(sch_msg));
    msg->pid = target_pid;
    
    double swap_s, swap_e;
    swap_s = what_time_is_it_now();

    DEBUG_PRINT(GREEN"Check Swap(%d)\n"RESET,target->priority);
    
    swapin(task_list, target);

    DEBUG_PRINT(GREEN"Swap Done(%d)\n"RESET,target->priority);

    swap_e = what_time_is_it_now();
#ifdef LOG
    fprintf(fps[target->priority-1],"%f,",(swap_e- swap_s));
#endif
    DEBUG_PRINT(GREEN"Scheduled Job(%d)\n"RESET,target->priority);    
    commErrchk(write(target->sch_dec_fd ,&ack,sizeof(int)));
}

void swapin(task_list_t * task_list, task_info_t *target){    
    int target_pid = target->pid;
    kill(target_pid, SIGUSR2);
    cudaAPI ret;
    do{
        ret = mm_request_handler(task_list, target);
    }while(ret != _Done_);
    
    // send to scheduler 
    target-> scheduled_time = what_time_is_it_now();
    int ack;
}



///// communication ////

int open_channel(char *pipe_name,int mode){
    int pipe_fd;
    
    if( access(pipe_name, F_OK) != -1)
        remove(pipe_name);

    if( mkfifo(pipe_name, 0666) == -1){
        DEBUG_PRINT(RED"[ERROR]Fail to make pipe"RESET);
        exit(-1);
    }
    if( (pipe_fd = open(pipe_name, mode)) < 0){
        DEBUG_PRINT(RED"[ERROR]Fail to open channel for %s\n"RESET, pipe_name);
        exit(-1);
    }
   DEBUG_PRINT(BLUE"Channel for %s has been successfully openned!\n"RESET, pipe_name);
   
   return pipe_fd;
}

void close_channel(char * pipe_name){
    if ( unlink(pipe_name) == -1){
        DEBUG_PRINT(RED"[ERROR]Fail to remove %s\n"RESET,pipe_name);
        exit(-1);
    }
}

void close_channels(task_info_t * task){
    char sch_req_fd_name[50];
    char sch_dec_fd_name[50];
    char mm_req_fd_name[50];
    char mm_dec_fd_name[50];

    snprintf(sch_req_fd_name, 50,"/tmp/sch_request_%d",task->pid);
    snprintf(sch_dec_fd_name, 50,"/tmp/sch_decision_%d",task->pid);
    snprintf(mm_req_fd_name, 50,"/tmp/mm_request_%d",task->pid);
    snprintf(mm_dec_fd_name, 50,"/tmp/mm_decision_%d",task->pid);
    
    close_channel(sch_req_fd_name);
    close_channel(sch_dec_fd_name);
    close_channel(mm_req_fd_name);
    close_channel(mm_dec_fd_name);
}

int make_fdset(fd_set *readfds, int reg_fd, task_list_t *task_list){
    int fd_head = 0;
    // initialize fd_set;
    FD_ZERO(readfds);

    // set register_fd
    FD_SET(reg_fd, readfds);
    if(reg_fd > fd_head) fd_head = reg_fd;
    // if there exist registered task, set
    if(task_list -> count > 0){
        task_info_t *node = task_list -> head;
        while(node != NULL){
            FD_SET(node -> sch_req_fd, readfds);
            FD_SET(node -> mm_req_fd, readfds);
            if(node -> sch_req_fd > fd_head) fd_head = node -> sch_req_fd;
            if(node -> mm_req_fd > fd_head) fd_head = node -> mm_req_fd;
            node = node -> next;
        }
    }
    return fd_head;
}


size_t swapout(task_list_t* proc_list, task_info_t* proc, size_t size){
    size_t evict_size = 0;
    list<int> evict_entry_list;

    // find evict pages
    auto iter  = proc->m_entry->begin();
    while(iter != proc->m_entry->end() && (evict_size <= size)){
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

    DEBUG_PRINT(GREEN "[SWAP OUT] Victim: %d, Size: %lu, Index: %d to %d\n" RESET, proc->pid, evict_size, evict_entry_front, evict_entry_back);

    commErrchk(write(proc->mm_dec_fd, msg, sizeof(evict_msg)));    
   
    kill(proc->pid, SIGUSR1);

    cudaAPI ret;
    do{
        ret = mm_request_handler(proc_list, proc);
    }while(ret != _Done_);

    return evict_size;
}


char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaFree_:
            return string(_cudaFree_);
        case _Done_:
            return string(_Done_);
        case _SWAPIN_:
            return string(_SWAPIN_);
    }
}