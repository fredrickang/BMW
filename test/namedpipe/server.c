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


#define BLUE "\x1b[34m" 
#define GREEN "\x1b[32m" 
#define RED "\x1b[31m"
#define RESET "\x1b[0m" 

#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[MMP][%s:%3d:%20s()]: " fmt, \
__FILE__, __LINE__, __func__, ##args);
#else
#define DEBUG_PRINT(fmt, args...)
#endif


int open_channel(char *pipe_name,int mode){
    int pipe_fd;
    
    if( access(pipe_name, F_OK) != -1)
        remove(pipe_name);

    if( mkfifo(pipe_name, 0666) == -1){
        DEBUG_PRINT(RED"[ERROR]Fail to make pipe\n"RESET );
        exit(-1);
    }
    DEBUG_PRINT("Pipe Made\n");
    if( (pipe_fd = open(pipe_name, mode)) < 0){
        DEBUG_PRINT(RED"[ERROR]Fail to open channel for %s\n"RESET , pipe_name);
        exit(-1);
    }
    DEBUG_PRINT(BLUE"Channel %s opened\n"RESET, pipe_name);
   
   return pipe_fd;
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

#define PATH "/tmp/test"

int main(void){
    int test_fd = open_channel(PATH, O_RDWR);
    DEBUG_PRINT("PATH open\n");
    return 0;
}