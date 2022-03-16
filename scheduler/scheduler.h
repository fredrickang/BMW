#ifndef SCHEDULER_H
#define SCHEDULER_H

void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
int main(int argc, char **argv);
double what_time_is_it_now();

typedef enum{
    IDLE, BUSY, ALIVE, TERM, WARMUP
}STATE;

typedef struct _TASK_INFO{
    int pid;
    int request_fd;
    int decision_fd;
    int priority;
    struct _TASK_INFO *next;
}task_info_t;

typedef struct _TASK_LIST{
    int count;
    task_info_t *head;
}task_list_t;

typedef struct node{
    int pid;
    int priority;
    struct node *next;
}node_t;

typedef struct queue{
    int count;
    node_t *front;
}queue_t;

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    int priority;
}reg_msg;

typedef struct _RESOURCE{
    STATE state;
    queue_t *waiting;
    int pid;
}resource_t;
#endif