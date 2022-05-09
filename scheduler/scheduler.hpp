
#include <stdio.h>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <string>

using namespace std;

void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
int main(int argc, char **argv);
double what_time_is_it_now();

typedef enum{
    IDLE, BUSY, ALIVE, TERM, WARMUP
}STATE;

typedef enum{
    _cudaMalloc_, _cudaFree_, _Done_, _SWAPIN_, _NotAPI_,
}cudaAPI;

typedef struct _TASK_INFO{
    int pid;
    int id;
    int sch_req_fd;
    int sch_dec_fd;
    int mm_req_fd;
    int mm_dec_fd;
    double scheduled_time;
    double period;
    double deadline;
    map<int,size_t> *m_entry;
    struct _TASK_INFO *next;
}task_info_t;

typedef struct _TASK_LIST{
    int count;
    task_info_t *head;
}task_list_t;

typedef struct node{
    int pid;
    double deadline;
    struct node *next;
}node_t;

typedef struct queue{
    int count;
    node_t *front;
}queue_t;

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    double period;
}reg_msg;

typedef struct _RESOURCE{
    STATE state;
    queue_t *waiting;
    int pid;
    double scheduled;
}resource_t;

typedef struct _MSG_PACKET_SCH{
    int pid;
}sch_msg;

typedef struct _MSG_PACKET_REQUEST{
    cudaAPI type;
    int entry_index;
    size_t size;
}req_msg;

typedef struct _MSG_PACKET_EVICT{
    int start_idx;
    int end_idx;
}evict_msg;

