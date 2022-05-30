#define _GNU_SOURCE

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
#include <sys/syscall.h>

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#define string(x) #x

//#define BLUE "\x1b[34m"  //info 
//#define GREEN "\x1b[32m" //highlight
//#define RED "\x1b[31m" // error
//#define RESET "\x1b[0m" 


#define BLUE  //info 
#define GREEN  //highlight
#define RED   // error
#define RESET  



#define gettid() syscall(SYS_gettid)

#define CHECK_COMM(ans) {check_comm((ans), __FILE__, __LINE__);}
inline void check_comm(int code, const char *file, int line, bool abort=true){
    if(code < 0){
        fprintf(stderr, RED "[customHook][%s:%3d]: [%d] CommError: %d\n" RESET,file,line, gettid(),code);
        if (abort) exit(code);
    }
}

#define CHECK_CUDA(ans) {check_cuda((ans), __FILE__, __LINE__);}
inline void check_cuda(int code, const char *file, int line, bool abort=true){
    if(code != 0){
        fprintf(stderr, RED "[customHook][%s:%3d]: [%d] CUDAERROR: %d\n" RESET,file,line, gettid(),code);
        if (abort) exit(code);
    }
}

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[customHook][%s:%3d:%30s()]: [%d] " fmt, \
__FILE__, __LINE__, __func__, gettid(), ##args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

#define REGISTRATION "/tmp/mmp"
#define REG_MSG_SIZE 2
#define REQ_MSG_SIZE 3
#define EVI_MSG_SIZE 2
#define SCH_MSG_SIZE 1

using namespace std;

static int init = 0;
static int entry_index = 0;
static int fake_address = -1;
static bool SWAP_OUT = false;

static map<void*, void*> pagetable;
pthread_t swap_thread_id;

typedef struct _ENTRY{
    void* address;
    size_t size;
}entry;

typedef struct _SWAP{
    void* origin_address;
    void* gpu_address;
    void* cpu_address;
    size_t size;
}gswap;

static  map<int,entry> gpu_entry_list;
static map<int,gswap> swap_entry_list;

int request_fd = -1;
int decision_fd = -1;

typedef enum{
    _cudaMalloc_, _cudaFree_, _Done_, _SWAPIN_
}cudaAPI;

typedef struct _MSG_PACKET_EVICT{
    int start_idx;
    int end_idx;
}evict_msg;

typedef struct _MSG_PACKET_IN{
    int type;
    size_t size;
}in_msg;

typedef struct _MSG_PACKET_REGIST{
    int reg_type;
    int pid;
}reg_msg;

typedef struct _MSG_PACKET_REQUEST{
    cudaAPI type;
    int entry_index;
    size_t size;
}req_msg;



void Init();
int SendRequest( void* devPtr, cudaAPI type, size_t size);
int SendRequest( void* devPtr, cudaAPI type, size_t size, int index);
char * getcudaAPIString(cudaAPI type);
void close_channels();
void close_channel(char * pipe_name);
void Cleanup();

void add_entry(map<int,entry>* entry_list, int index,  void* devPtr, size_t size);
void del_entry(map<int,entry>* entry_list,  void* devPtr);

void add_swap_entry(map<int,gswap>* entry_list, int index, void* origPtr, void* gpuPtr,  void* cpuPtr, size_t size);
void del_swap_entry(map<int,gswap>* entry_list,  void* devPtr);

int find_index_by_ptr(map<int,entry>* entry_list,  void* devPtr);
bool exist_in_entry(map<int,entry> * entry_list, void *devPtr);
int floorSearch(void * addr);
void swapout(int signum);
void swapin(int signum);
void DEBUG_PRINT_SWAP();
void DEBUG_PRINT_PAGETABLE();
void DEBUG_PRINT_ENTRY();
void * check_pointer_arithmetic(void *, const char*);
float checksum(float * input, int size);

/* CUDA memory hook */
static cudaError_t (*lcudaMalloc)(void **, size_t) = (cudaError_t (*) (void**, size_t))dlsym(RTLD_NEXT,"cudaMalloc");
static cudaError_t (*lcudaFree) (void*) = (cudaError_t (*) (void *))dlsym(RTLD_NEXT,"cudaFree");
static cudaError_t (*lcudaMemcpy) (void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) (void*, const void*, size_t, cudaMemcpyKind))dlsym(RTLD_NEXT, "cudaMemcpy");
static cudaError_t (*lcudaLaunchKernel) (const void*, dim3, dim3, void**, size_t, cudaStream_t) = (cudaError_t (*) (const void*, dim3, dim3, void**, size_t, cudaStream_t))dlsym(RTLD_NEXT, "cudaLaunchKernel");
static cudaError_t (*lcudaMemsetAsync) (void*, int, size_t, cudaStream_t) = (cudaError_t (*) (void*, int, size_t, cudaStream_t))dlsym(RTLD_NEXT, "cudaMemsetAsync");
static cublasStatus_t (*lcublasSgemm) (cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) = (cublasStatus_t (*) (cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int))dlsym(RTLD_NEXT,"cublasSgemm_v2");
