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

#include "scheduler_fn.h"


#define REG_MSG_SIZE 3

#define BLUE "\x1b[34m" //info
#define GREEN "\x1b[32m" // highlight
#define RED "\x1b[31m" // error
#define RESET "\x1b[0m" 

extern int mmp2sch_fd;
extern int sch2mmp_fd;

#define commErrchk(ans) {commAssert((ans), __FILE__, __LINE__);}
inline void commAssert(int code, const char *file, int line){
    if(code < 0){
        fprintf(stderr, RED"[scheduler][%s:%3d]: CommError: %d\n"RESET,file,line,code);
        exit(code);
    }
}

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[scheduler][%s:%3d:%20s()]: " fmt, \
__FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINT(fmt, args...) 
#endif

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

int dequeue(queue_t *q, double current_time, resource_t *res){
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
    
    q -> count --;
    free(target);
    print_queue("WQ", q);
    return target_pid;
}  

void send_release_time(task_list_t *task_list, double current_time){
    struct timespec release_time;
    clock_gettime(CLOCK_MONOTONIC, &release_time);

    for(task_info_t * node = task_list -> head ; node != NULL ; node = node -> next){
        if(write(node->decision_fd, &release_time,sizeof(struct timespec)) < 0)
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
        
    while( read(reg_fd, msg, REG_MSG_SIZE*sizeof(int)) > 0){
        if(msg -> regist == 1) 
            do_register(task_list, msg); 
        else 
            deregister(task_list, msg, res);
    }
}

void do_register(task_list_t *task_list, reg_msg *msg){
    task_info_t *task = (task_info_t *)malloc(sizeof(task_info_t));
    task -> pid = msg -> pid;
    task -> priority = msg -> priority;

    DEBUG_PRINT(BLUE"======== REGISTRATION ========\n"RESET);
    DEBUG_PRINT(BLUE"[PID]      %3d\n"RESET, task-> pid);
    DEBUG_PRINT(BLUE"[Priority] %3d\n"RESET, task->priority);
    
    char req_fd_name[50];
    char dec_fd_name[50];

    snprintf(req_fd_name, 50,"/tmp/sch_request_%d",task->pid);
    snprintf(dec_fd_name, 50,"/tmp/sch_decision_%d",task->pid);

    task -> request_fd = open_channel(req_fd_name, O_RDONLY);
    task -> decision_fd = open_channel(dec_fd_name, O_WRONLY);
    
    task -> next = NULL;

    register_task(task_list, task);    
}


void deregister(task_list_t *task_list, reg_msg *msg, resource_t *res){
    int pid;
    task_info_t *target = find_task_by_pid(task_list, msg -> pid);
    pid = target -> pid;
    close_channels(target);
    de_register_task(task_list, target);
        
    if (res-> pid == pid) res->state = IDLE;
}

// Request Handler //

void request_handler(task_list_t *task_list, task_info_t *task, resource_t *res, double current_time){    
    int ack;
    commErrchk(read(task -> request_fd, &ack, sizeof(int)*1));

    if( res -> state == BUSY && res -> pid == task->pid){ /* Job termniation */
        DEBUG_PRINT(GREEN"Term Job(%d)\n"RESET,task->priority);
        res -> state = IDLE;
        res -> pid = -1;
    }
    else{ /* Job release */
        DEBUG_PRINT(GREEN"Release Job(%d)\n"RESET,task->priority);
        enqueue(res->waiting, task->pid, task->priority);
    }

}

void decision_handler(int target_pid, task_list_t *task_list){

    int ack = 0;
    task_info_t *target = find_task_by_pid(task_list, target_pid);

    //
    sch_msg * msg = (sch_msg *)malloc(sizeof(sch_msg));
    msg->pid = target_pid;
    
    DEBUG_PRINT(GREEN"Check Swap(%d)\n"RESET,target->priority);
    commErrchk(write(sch2mmp_fd, msg, sizeof(int)*1));
    commErrchk(read(mmp2sch_fd, &ack, sizeof(int)));
    DEBUG_PRINT(GREEN"Swap Done(%d)\n"RESET,target->priority);

    DEBUG_PRINT(GREEN"Scheduled Job(%d)\n"RESET,target->priority);    
    commErrchk(write(target->decision_fd,&ack,sizeof(int)));
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
    char request_name[30];
    char decision_name[30];
    
    snprintf(request_name, 30, "/tmp/sch_request_%d", task->pid);
    snprintf(decision_name, 30, "/tmp/sch_decision_%d", task->pid);
    
    close_channel(request_name);
    close_channel(decision_name);
}

int make_fdset(fd_set *readfds,int reg_fd, task_list_t *task_list){
    // initialize fd_set;
    FD_ZERO(readfds);

    // set register_fd
    FD_SET(reg_fd, readfds);
        
    // if there exist registered task, set
    if(task_list -> count > 0){
        task_info_t *node = task_list -> head;
        while(node != NULL){
            FD_SET(node -> request_fd, readfds);
            node = node -> next;
        }
        return task_list -> head ->request_fd;
    }
    return reg_fd;
}
