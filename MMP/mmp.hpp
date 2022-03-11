#include <stdio.h>
#include <iostream>
#include <map>
#include <list>
#include <vector>
#include <string>

using namespace std;

typedef enum{
    _cudaMalloc_, _cudaMemcpy_, _cudaMemcpyAsync_, _cudaFree_
}cudaAPI;

typedef enum {
    _cudaMemcpyHostToHost_, _cudaMemcpyHostToDevice_, _cudaMemcpyDeviceToHost_, _cudaMemcpyDeviceToDevice_, _cudaMemcpyDefault_
}cudaMemcpyKind;

typedef struct _PROC_INFO{
    int pid;
    int id;
    map<int,size_t> *m_entry;
    map<int,size_t> *swap_entry;
    int request_fd;
    int decision_fd;
    struct _PROC_INFO *next;
}_proc;

typedef struct _PROC_LIST{
    int count;
    _proc *head;
}_proc_list;

typedef struct _MSG_PACKET_REGIST{
    int reg_type;
    int pid;
}reg_msg;

typedef struct _MSG_PACKET_REQUEST{
    cudaAPI type; // Memory related function type (e.g. cudaMalloc, cudaMemcpy ...etc)
    cudaMemcpyKind kind; // cudaMemcpy kind, if not cudamemcpy -1
    int entry_index;
    int size; // Memory request size
}req_msg;

typedef struct _MSG_PACKET_EVICT{
    int start_idx;
    int end_idx;
}evict_msg;