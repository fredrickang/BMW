import argparse
import subprocess
from threading import Thread
import os
import signal
import time
import datetime
from tqdm import tqdm


def scheduler(num_tasks, exp_log_dir):
    command_line = []
#    command_line.append("gdb")
#    command_line.append("-ex=r")
#    command_line.append("--args")
    command_line.append("/home/xavier5/BMW/scheduler/scheduler")
    command_line.append("-sync")
    command_line.append(str(num_tasks))
    command_line.append("-logdir")
    command_line.append(exp_log_dir)
    
    #sub = subprocess.Popen(command_line)
 
    sub = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = sub.communicate()
    log_path = os.path.join(exp_log_dir,"scheduler")
    fp = open(log_path+"/stderr.log",'w')
    fp.write(stderr.decode('ascii'))
    fp.close()

    
def scheduler_wo_mmp(num_tasks, exp_log_dir):
    command_line = []
#    command_line.append("gdb")
#    command_line.append("-ex=r")
#    command_line.append("--args")
    print("scheduler wo mmp")
    command_line.append("/home/xavier5/BMW/scheduler/scheduler_wo_mmp")
    command_line.append("-sync")
    command_line.append(str(num_tasks))
    command_line.append("-logdir")
    command_line.append(exp_log_dir)
    #sub = subprocess.Popen(command_line)
    
    sub = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = sub.communicate()
    log_path = os.path.join(exp_log_dir,"scheduler")
    fp = open(log_path+"/stderr.log",'w')
    fp.write(stderr.decode('ascii'))
    fp.close()


def mmp(exp_log_dir):
    command_line = []
    #command_line.append("gdb")
    #command_line.append("-ex=r")
    #command_line.append("--args")
    command_line.append("/home/xavier5/BMW/MMP/mmp")
    #sub = subprocess.Popen(command_line)
    
    sub = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = sub.communicate()
    log_path = os.path.join(exp_log_dir,"mmp")
    fp = open(log_path+"/stderr.log",'w')
    fp.write(stderr.decode('ascii'))
    fp.close()


def dnn(prio, wMMP, exp_log_dir):
    prio = prio+1

    command_line = []
    command_line.append("./darknet")
    command_line.append("detector")
    command_line.append("test")
    command_line.append("cfg/coco.data")
    command_line.append("cfg/yolov3_p{:d}.cfg".format(prio))
    command_line.append("yolov3.weights")
    command_line.append("data/dog.jpg")
    command_line.append("-logdir")
    command_line.append(exp_log_dir)

    env = os.environ
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
    if wMMP == 1:
        env["LD_PRELOAD"] = "/home/xavier5/BMW/libcuhooklib.so"

    #sub = subprocess.Popen(command_line, env=env)

    sub = subprocess.Popen(command_line, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = sub.communicate()
    sub.wait()
    log_path = os.path.join(exp_log_dir,"darknet")
    fp = open(log_path+"/stderr_{:d}.log".format(prio),'w')
    fp.write(stderr.decode('ascii'))
    fp.close()
    ret = sub.returncode
    print("YOLO{:d} ret: {:d}".format(prio, ret))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--wMMP", type=int, default=1)
    parser.add_argument("--wSCH", type=int, default=1)

    parser.add_argument("--numtasks", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)

    parser.add_argument("--logdir", type=str, default="/home/xavier5/BMW/logs")

    opt = parser.parse_args()
    
    print(opt)
    
    # mk log dir
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    exp_detail = "nt{}tri{}M{}S{}".format(opt.numtasks,opt.trials,opt.wMMP,opt.wSCH)
    exp_full_name = "{}_{}".format(exp_detail,current_time)
    exp_log_dir = os.path.join(opt.logdir,exp_full_name)
    if not os.path.isdir(exp_log_dir):
        os.mkdir(exp_log_dir)
        yolo_log_path = os.path.join(exp_log_dir, "darknet")
        sch_log_path = os.path.join(exp_log_dir, "scheduler")
        mmp_log_path = os.path.join(exp_log_dir, "mmp")
        os.mkdir(yolo_log_path)
        os.mkdir(sch_log_path)
        os.mkdir(mmp_log_path)

    for i in tqdm(range(opt.trials)):
        task_threads =[]
        for i in range(opt.numtasks):
            args = [i]
            if opt.wMMP == 1:
                args.append(1)
            else:
                args.append(0)
            args.append(exp_log_dir)
            
            task_threads.append(Thread(target = dnn, args = args))
        
        with open("/proc/sys/vm/drop_caches","w") as stream:
            stream.write("3\n")
        
        for thread in task_threads:
            thread.start()
        
        for thread in task_threads:
            thread.join()
