#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <map>

#define REGISTRATION "/tmp/registration"
#define string(x) #x

#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[customHook][%s:%3d:%20s()]: [%d] " fmt, \
__FILE__, __LINE__, __func__,getpid(), ##args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

using namespace std;

typedef struct _ENTRY{
    const void * address;
    size_t size;
}entry;

static int entry_index = 0;
static map<int,entry> gpu_m_entry;
static map<int,const void*> cpu_m_entry;
static int init = 0;
int request_fd = -1;
int decision_fd = -1;
int register_fd = -1;



typedef enum{
    _cudaMalloc_, _cudaMemcpy_, _cudaMemcpyAsync_, _cudaFree_
}cudaAPI;

typedef struct _MSG_PACKET_EVICT{
    int start_idx;
    int end_idx;
}evict_msg;

typedef struct _MSG_PACKET_REGIST{
    int reg_type;
    int pid;
}reg_msg;

typedef struct _MSG_PACKET_REQUEST{
    cudaAPI type;
    cudaMemcpyKind kind;
    int entry_index;
    int size;
}req_msg;


void Init();
int SendRequest(const void* devPtr, cudaAPI type, cudaMemcpyKind kind, size_t size);
char * getcudaAPIString(cudaAPI type);
char * getcudaMemcpyKindString(cudaMemcpyKind kind);
void close_channels();
void close_channel(char * pipe_name);
void Cleanup();

void Add_entry(const void* devPtr);
void Del_entry(const void* devPtr);
int find_index_by_ptr(const void* devPtr);

void sigint(int signum);
char * getcudaAPIString(cudaAPI type);
char * getcudaMemcpyKindString(cudaMemcpyKind kind);
void DEBUG_PRINT_ENTRY();


/* CUDA memory hook */
static cudaError_t (*lcudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*lcudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind) = NULL;
static cudaError_t (*lcudaMemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = NULL;
static cudaError_t (*lcudaFree) (void*) = NULL;

cudaError_t cudaMalloc (void **devPtr, size_t size){   
    cudaError_t err;
    if(!init){
        Init();
        init = 1;
    }

    cudaError_t (*lcudaMalloc) (void **, size_t) = (cudaError_t (*) (void**, size_t))dlsym(RTLD_NEXT,"cudaMalloc");
    DEBUG_PRINT("cudaMalloc [%d]\n", size);

    SendRequest((const void *)*devPtr, _cudaMalloc_, cudaMemcpyHostToHost, size);
    err = lcudaMalloc(devPtr, size);
    Add_entry((const void *)*devPtr, size);
    return err;
}


cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind){
    cudaError_t err;

    cudaError_t (*lcudaMemcpy) (void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void*, const void*, size_t, cudaMemcpyKind))dlsym(RTLD_NEXT, "cudaMemcpy");
    DEBUG_PRINT("cudaMemcpy-%s [%d]\n",getcudaMemcpyKindString(kind),count);

    if(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
        SendRequest((const void *)dst, _cudaMemcpy_, kind, count);
        err = lcudaMemcpy(dst, src, count, kind );
        Add_entry((const void *)dst, count);
    }
    if(kind == cudaMemcpyDeviceToHost){
       SendRequest((const void *)src, _cudaMemcpy_, kind, count); 
       err = lcudaMemcpy(dst, src, count, kind );
       Del_entry((const void *)src);
    }
    
    return err;
}

cudaError_t cudaMemcpyAsync (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str){
    cudaError_t err;

    cudaError_t (*lcudaMemcpyAsync) (void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    DEBUG_PRINT("cudaMemcpyAsync-%s [%d]\n",getcudaMemcpyKindString(kind),count);
        
    if(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) {
        SendRequest((const void *)dst, _cudaMemcpyAsync_, kind, count);
        lcudaMemcpyAsync(dst, src, count, kind, str );
        Add_entry((const void *)dst, count);
    }
    if(kind == cudaMemcpyDeviceToHost){
       SendRequest((const void *)src, _cudaMemcpyAsync_, kind, count); 
       lcudaMemcpyAsync(dst, src, count, kind, str );
       Del_entry((const void *)src);
    }  

    return err;
}

cudaError_t cudaFree(void* devPtr){ /* free */
    

    cudaError_t (*lcudaFree) (void*) = (cudaError_t (*) (void *))dlsym(RTLD_NEXT,"cudaFree");
    DEBUG_PRINT("cudaFree\n");
    SendRequest((const void *)devPtr, _cudaFree_, cudaMemcpyHostToHost, 0);
    Del_entry((const void *)devPtr);

    return lcudaFree(devPtr);
}

void Init(){
    signal(SIGINT, sigint);
    if((register_fd = open(REGISTRATION, O_WRONLY)) < 0){
        DEBUG_PRINT("\x1b[31m""REGISTRATION CHANNEL OPEN FAIL\n""\x1b[0m");
        exit(-1);
    }

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 1;
    reg->pid = getpid();

    if(write(register_fd, reg, sizeof(int)*2) < 0){
        DEBUG_PRINT("\x1b[31m""REGISTRATION FAIL\n""\x1b[0m");
        exit(-1);
    }

    DEBUG_PRINT("Registrated\n");

    char request[30];
    char decision[30];

    snprintf(request, 30, "/tmp/request_%d",getpid());
    snprintf(decision, 30, "/tmp/decision_%d",getpid());

    while((request_fd = open(request,O_WRONLY)) < 0);
    while((decision_fd = open(decision,O_RDONLY)) < 0);
    DEBUG_PRINT("Request/Decision channel opened\n");

    atexit(Cleanup);
    DEBUG_PRINT("Termination function registered\n");
    
    DEBUG_PRINT_ENTRY();
}

int SendRequest(const void* devPtr, cudaAPI type, cudaMemcpyKind kind, size_t size){
    DEBUG_PRINT_ENTRY();
    req_msg * msg = (req_msg *)malloc(sizeof(req_msg));
    
    msg -> type = type;
    msg -> kind = kind;
    msg -> size = size;
    
    if(type == _cudaMalloc_ || ((type == _cudaMemcpy_ || type == _cudaMemcpyAsync_) && (kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice))) \
        msg -> entry_index = entry_index;
    
    if(type == _cudaFree_ || ((type == _cudaMemcpy_ || type == _cudaMemcpyAsync_) && kind == cudaMemcpyDeviceToHost)) \
        msg -> entry_index = find_index_by_ptr(devPtr);

    if(write(request_fd, msg, sizeof(int)*4) < 0){
        DEBUG_PRINT("\x1b[31m""SendRequest WRITE fail [%s %s %d]\n""\x1b[0m" ,getcudaAPIString(type), getcudaMemcpyKindString(kind), size );
        exit(-1);
    }
    // DEBUG_PRINT("Request sent [%s %s %d]\n",getcudaAPIString(type), getcudaMemcpyKindString(kind), size);
    int ack;
    if(read(decision_fd, &ack, sizeof(int)) < 0){
        DEBUG_PRINT("\x1b[31m""SendRequest READ fail [%s %s %d]\n""\x1b[0m" , getcudaAPIString(type), getcudaMemcpyKindString(kind), size );
        exit(-1);
    }
    // DEBUG_PRINT("Decision Ack\n");
}

void DEBUG_PRINT_ENTRY(){
    DEBUG_PRINT("Current Entry: ");
    auto iter = gpu_m_entry.begin();
    while(iter != gpu_m_entry.end()){
        fprintf(stderr, "{%d, [%p, %d]} ",iter->first, iter->second.address, iter->second.size);
        ++iter;
    }
    fprintf(stderr,"\n");
}

void Add_entry(const void* devPtr, size_t size){
    //DEBUG_PRINT_ENTRY();
    DEBUG_PRINT("Add entry: {%d, %p}\n", entry_index, devPtr);
    entry tmp = {devPtr, size};
    gpu_m_entry.insert({entry_index, tmp});
    entry_index++;
    //DEBUG_PRINT_ENTRY();
}

void Del_entry(const void* devPtr){
    //DEBUG_PRINT_ENTRY();
    DEBUG_PRINT("Del entry: %p\n", devPtr);
    gpu_m_entry.erase(find_index_by_ptr(devPtr));
    //DEBUG_PRINT_ENTRY();
}

int find_index_by_ptr(const void* ptr){
    //DEBUG_PRINT_ENTRY();
    auto iter = gpu_m_entry.begin();
    
    while(iter != gpu_m_entry.end() && iter->second.address != ptr ){
        ++iter;
    }

    if(iter == gpu_m_entry.end() && iter->second.address != ptr) {
        DEBUG_PRINT("\x1b[31m""Can't find ptr inside entry\n""\x1b[0m");
        exit(-1);
    }
    
    return iter->first;
}


void Cleanup(){
    DEBUG_PRINT("Cleaning up...\n");

    reg_msg *reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->reg_type = 0;
    reg->pid = getpid();
    if(write(register_fd, reg, sizeof(int)*2) < 0){
        DEBUG_PRINT("\x1b[31m""DE-REGISTRATION FAIL\n""\x1b[0m");
        exit(-1);
    }
    DEBUG_PRINT(" De-registration done\n");

}

void sigint(int signum){
    signal(SIGINT, sigint);
    DEBUG_PRINT("\x1b[31m""[%d] SIGINT handler callback\n""\x1b[0m", getpid());

    evict_msg *msg = (evict_msg *)malloc(sizeof(evict_msg));
    read(decision_fd, msg, sizeof(int)*2); 

    auto begin_iter = gpu_m_entry.find(msg->start_idx);
    auto end_iter = gpu_m_entry.find(msg->end_idx);
    
    for(auto iter = begin_iter; iter!=end_iter; iter++){
        int index = iter->first;
        

    }
    

}

char * getcudaAPIString(cudaAPI type){
    switch (type){
        case _cudaMalloc_:
            return string(_cudaMalloc_);
        case _cudaMemcpy_:
            return string(_cudaMemcpy_);
        case _cudaMemcpyAsync_:
            return string(_cudaMemcpyAsync_);
        case _cudaFree_:
            return string(_cudaFree_);
    }
}

char * getcudaMemcpyKindString(cudaMemcpyKind kind){
    switch (kind){
        case cudaMemcpyHostToHost:
            return string(cudaMemcpyHostToHost);
        case cudaMemcpyHostToDevice:
            return string(cudaMemcpyHostToDevice);
        case cudaMemcpyDeviceToHost:
            return string(cudaMemcpyDeviceToHost);
        case cudaMemcpyDeviceToDevice:
            return string(cudaMemcpyDeviceToDevice);
        case cudaMemcpyDefault:
            return string(cudaMemcpyDefault);
    }
}