#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[MMP][%s:%3d:%20s()]: " fmt, \
__FILE__, __LINE__, __func__, ##args);
#else
#define DEBUG_PRINT(fmt, args...)
#endif


void del_arg(int argc, char **argv, int index);
int find_int_arg(int argc, char **argv, char *arg, int def);
double what_time_is_it_now();

int make_fdset(fd_set *readfds,int reg_fd, _proc_list *proc_list);
_proc_list *create_proc_list();
void check_registration(_proc_list *proc_list, int reg_fd);
void registration(_proc_list* proc_list, reg_msg *msg);
void de_registration(_proc_list* proc_list, reg_msg *msg);
void de_register_proc(_proc_list* proc_list, _proc * target);
void request_handler(_proc_list * proc_list, _proc * proc);
_proc *find_proc_by_pid(_proc_list *proc_list, int pid);
int open_channel(char *pipe_name,int mode);
char * getcudaAPIString(cudaAPI type);
void close_channel(int pid, char * pipe_name);
void close_channels(int pid);
_proc* choose_victim(_proc_list* proc_list, _proc* proc);
void evictprotocal(_proc* proc, size_t size);
