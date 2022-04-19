#ifndef SCHEDULER_FN_H
#define SCHEDULER_FN_H

#ifdef DEBUG
  #define DEBUGP 1
  #else 
  #define DEBUGP 0
#endif 

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[scheduler][%s:%3d:%20s()]: " fmt, \
__FILE__, __LINE__, __func__, ##args)
#endif


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

#include "scheduler.h"
void set_priority(int priority);
void set_affinity(int core);

task_list_t *create_task_list();
void register_task(task_list_t *task_list, task_info_t *task);
void de_register_task(task_list_t *task_list, task_info_t *task); 
resource_t *create_resource();
node_t* new_node(int pid,int priority);
queue_t *create_queue();
int enqueue(queue_t *q, int pid, int priority);
void nodeDelete(queue_t *q, node_t *del);
int dequeue(queue_t *q, double current_time, resource_t *res);
void update_deadline(task_info_t * task, double current_time);
void send_release_time(task_list_t *task_list, double current_time);
task_info_t *find_task_by_id(task_list_t *task_list, int id);
task_info_t *find_task_by_pid(task_list_t *task_list, int pid);
void print_list(char * name, task_list_t * task_list);
void print_queue(char * name, queue_t * q);
void check_registration(task_list_t *task_list, int reg_fd, resource_t *res);
void do_register(task_list_t *task_list, reg_msg *msg);
void deregister(task_list_t *task_list, reg_msg *msg, resource_t *res);
void request_handler(task_list_t *task_list, task_info_t *task, resource_t *res, resource_t*, double current_time);    
void decision_handler(int target_pid, task_list_t *task_list);
void init_decision_handler(int target_pid, task_list_t *task_list);
int enqueue_backward(queue_t *q, int pid, int priority);

int open_channel(char *pipe_name,int mode);
void close_channel(char * pipe_name);
void close_channels(task_info_t * task);
int make_fdset(fd_set *readfds,int reg_fd, task_list_t *task_list);

#endif