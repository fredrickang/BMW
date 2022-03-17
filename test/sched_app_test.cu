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

#define MAX_ITER 5
#define MAX_JOB 30

int request_fd = -1;
int decision_fd = -1;
int register_fd = -1;
struct timespec release_time = {0,0};

typedef struct _MSG_PACKET{
    int regist;
    int pid;
    int priority;
}msg;

int Sync = 1;

int communicate(int ack){
    int decision = 0; 
    
    if( write(request_fd, &ack, sizeof(int)*1) == -1){
        perror("Request Send :");
        exit(-1);
    }
    if(Sync){
        if(read(decision_fd, &release_time, sizeof(struct timespec))< 0){
            perror("release time");
        }
        Sync = 0;
    }

    if(read(decision_fd, &decision, sizeof(int)*1) == -1){
        perror("Decision Recv :");
        exit(-1);
    }
    printf("decision: %d\n",decision);
    return decision;
}

void timespec_add (struct timespec *left,
              const struct timespec *right)
{
  left->tv_sec = left->tv_sec + right->tv_sec;
  left->tv_nsec = left->tv_nsec + right->tv_nsec;
  while (left->tv_nsec >= 1000000000)
    {
      ++left->tv_sec;
      left->tv_nsec -= 1000000000;
    }
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

void cleanup(void){
    msg * dummy = (msg *)malloc(sizeof(msg));
    dummy->regist = 0;
    dummy->pid = getpid();
    write(register_fd, dummy, sizeof(int)*3);
}

int main(int argc, char **argv){
    atexit(cleanup);
    int period = find_int_arg(argc, argv, "-period", 500);
    int priority = find_int_arg(argc, argv, "-prio", 1);

    double period_ns = period*1000000;
    struct timespec period_st = {0, 0};
    period_st.tv_nsec = period_ns;
    while(period_st.tv_nsec >= 1000000000){
        ++period_st.tv_sec;
        period_st.tv_nsec -= 1000000000;
    }
    /*   REGISTRATION   */

    if( (register_fd = open("/tmp/scheduler",O_WRONLY))< 0){
        perror("Opening Registration : ");
        exit(-1);
    }

    msg * dummy = (msg *)malloc(sizeof(msg));
    dummy->regist = 1;
    dummy->pid = getpid();
    dummy->priority = priority;

    if(write(register_fd, dummy, 3*sizeof(int)) < 0){
        perror("Registrating : ");
        exit(-1);
    }
    printf("==%d== Registrated!\n",getpid());

    char request[50];
    char decision[50];

    snprintf(request, 50, "/tmp/request_%d",getpid());
    snprintf(decision, 50, "/tmp/decision_%d",getpid());

    while( (request_fd = open(request, O_WRONLY)) < 0);
    while( (decision_fd = open(decision, O_RDONLY)) < 0);
    printf("==%d== comms open!\n",getpid());
    /*   !REGISTRATION   */
    


    int *a[MAX_ITER];
    int *d_a[MAX_ITER];
    for(int i = 0; i < MAX_ITER; i++){
        a[i] = (int *)malloc(sizeof(int)*1000);
        for(int j = 0; j < 1000; j++) a[i][j] = i;
        cudaMalloc(&d_a[i],sizeof(int)*1000);
        cudaMemcpy(d_a[i],a[i], sizeof(int)*1000, cudaMemcpyHostToDevice);
    }

    int *b[MAX_ITER];
    int ret;
    int ack;
    for(int i = 0; i < MAX_JOB; i++){
        communicate(0);

        printf("======= JOB %d =======\n",i);

        for(int j = 0; j < MAX_ITER; j++){
            b[j] = (int *)malloc(sizeof(int)*1000);
            ret = cudaMemcpy(b[j],d_a[j], sizeof(int)*1000, cudaMemcpyDeviceToHost);
            printf("Index %d, ret:%d, values: %d %d %d\n",j, ret, b[j][0], b[j][1], b[j][2]);
        }
        
        if( write(request_fd, &ack, sizeof(int)*1) == -1){
            perror("Request Send :");
            exit(-1);
        }
        
        struct timespec current;
        clock_gettime(CLOCK_MONOTONIC, &current);
        timespec_add(&release_time, &period_st);
        clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&release_time, NULL);
    } 

}