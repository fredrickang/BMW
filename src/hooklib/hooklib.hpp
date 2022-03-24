#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <map>
#include <list>
#include <pthread.h>

#define BLUE "\x1b[34m"  //info 
#define GREEN "\x1b[32m" //highlight
#define RED "\x1b[31m" // error
#define RESET "\x1b[0m" 

#define DEBUG
#define commErrchk(ans) {commAssert((ans), __FILE__, __LINE__);}
inline void commAssert(int code, const char *file, int line, bool abort=true){
    if(code < 0){
        fprintf(stderr, RED "[customHook][%s:%3d]: [%d] CommError: %d\n" RESET,file,line, getpid(),code);
        if (abort) exit(code);
    }
}

#define CHECK_CUDA(ans) {check_cuda((ans), __FILE__, __LINE__);}
inline void check_cuda(int code, const char *file, int line, bool abort=true){
    if(code != 0){
        fprintf(stderr, RED "[customHook][%s:%3d]: [%d] CUDAERROR: %d\n" RESET,file,line, getpid(),code);
        if (abort) exit(code);
    }
}


#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) do{ fprintf(stderr, "[customHook][%s:%3d:%20s()]: [%d] " fmt, \
__FILE__, __LINE__, __func__, getpid(), ##args); fflush(stderr); }while(0)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

#define REGISTRATION "/tmp/mmp"
#define string(x) #x

#define REG_MSG_SIZE 2
#define REQ_MSG_SIZE 3
#define EVI_MSG_SIZE 2
#define SCH_MSG_SIZE 1

#define GPU_PAGE_SIZE 512

using namespace std;


typedef struct _ENTRY{
    void** address;
    size_t size;
}entry;

typedef struct _SWAP{
    void** gpu_address;
     void* cpu_address;
    size_t size;
}gswap;

static int init = 0;

static bool SWAP_OUT = false;

static int entry_index = 0;
static  map<int,entry> gpu_entry_list;
static map<int,gswap> swap_entry_list;

int request_fd = -1;
int decision_fd = -1;
int register_fd = -1;

pthread_t swap_thread_id;

typedef enum{
    _cudaMalloc_, _cudaFree_, _Done_
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
    int entry_index;
    int size;
}req_msg;



void Init();
int SendRequest( void* devPtr, cudaAPI type, size_t size);
int SendRequest( void* devPtr, cudaAPI type, size_t size, int index);
char * getcudaAPIString(cudaAPI type);
void close_channels();
void close_channel(char * pipe_name);
void Cleanup();

void add_entry(map<int,entry>* entry_list, int index,  void** devPtr, size_t size);
void del_entry(map<int,entry>* entry_list,  void* devPtr);

void add_swap_entry(map<int,gswap>* entry_list, int index,  void** gpuPtr,  void* cpuPtr, size_t size);
void del_swap_entry(map<int,gswap>* entry_list,  void* devPtr);

int find_index_by_ptr(map<int,entry>* entry_list,  void* devPtr);

void swapout(int signum);
void swapin(int signum);
void DEBUG_PRINT_ENTRY();

/* CUDA memory hook */
static cudaError_t (*lcudaMalloc)(void **, size_t) = (cudaError_t (*) (void**, size_t))dlsym(RTLD_NEXT,"cudaMalloc");
static cudaError_t (*lcudaFree) (void*) = (cudaError_t (*) (void *))dlsym(RTLD_NEXT,"cudaFree");