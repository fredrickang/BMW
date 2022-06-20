#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <float.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>



// Allocate 30GB memory in 4 bytes 




int request_fd = -1;
int decision_fd = -1;
int register_fd = -1;
int Sync = 1;
struct timespec *release_time = NULL;

typedef struct _MSG_PACKET_REG{
    int regist;
    int pid;
    double period;
}reg_msg;

int communicate(int ack){
    int decision = 0; 
    
    if( write(request_fd, &ack, sizeof(int)*1) == -1){
        perror("Request Send :");
        exit(-1);
    }
    if(Sync){
        release_time = (struct timespec *)malloc(sizeof(struct timespec));
        if(read(decision_fd, release_time, sizeof(struct timespec))< 0){
            perror("release time");
        }
        Sync = 0;
    }

    if(read(decision_fd, &decision, sizeof(int)*1) == -1){
        perror("Decision Recv :");
        exit(-1);
    }
    return decision;
}

int main(void){
    
    if( (register_fd = open("/tmp/scheduler", O_WRONLY)) < 0){
        perror("Opening Registration channel");
        exit(-1);
    }

    reg_msg * reg = (reg_msg *)malloc(sizeof(reg_msg));
    reg->regist = 1;
    reg->pid = getpid();
    reg->period = 100;

    if(write(register_fd, reg, sizeof(reg_msg)) < 0){
        perror("Registrating: ");
        exit(-1);
    }

    fprintf(stderr, "Registrated (%d)\n", getpid());

    char request[50];
    char decision[50];

    snprintf(request, 50, "/tmp/sch_request_%d",getpid());
    snprintf(decision, 50, "/tmp/sch_decision_%d",getpid());

    while( (request_fd = open(request, O_WRONLY)) < 0);
    while( (decision_fd = open(decision, O_RDONLY)) < 0);
    fprintf(stderr, "==%d== comms open!\n",getpid());

    void *tmp;
    cudaMalloc(&tmp, 1);
    int ack = 99;
    if( write(request_fd, &ack, sizeof(int)*1) == -1){
        perror("Request Send :");
        exit(-1);
    }
    if(read(decision_fd, &ack, sizeof(int)*1) == -1){
        perror("Decision Recv :");
        exit(-1);
    }

    communicate(0);

    int *** chunks = (int ***)malloc(sizeof(int **)*10000);
    for(int i = 0; i < 10000; i++){
        chunks[i] = (int **)malloc(sizeof(int *)*10);
        for(int j = 0; j < 10; j++){
            cudaMalloc(&chunks[i][j], sizeof(int)*2560);
        }
    }
    
    if (write(request_fd, &ack, sizeof(int)) == -1){
        perror("Request Send:");
        exit(-1);
    }

    communicate(0);
}