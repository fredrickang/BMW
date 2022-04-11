MMP="/home/xavier5/BMW/MMP/mmp"
SCHEDULER="/home/xavier5/BMW/scheduler/scheduler"
DARKNET="/home/xavier5/BMW/darknet"
CUDAHOOK="/home/xavier5/BMW/libcuhooklib.so"

TASKNUM=1

echo "xavier5" | sudo -S $MMP &
echo "xavier5" | sudo -S $SCHEDULER -sync $TASKNUM &
echo "xavier5" | sudo -S LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=$CUDAHOOK $DARKNET/darknet detector test $DARKNET/cfg/coco.data $DARKNET/cfg/yolov3_p1.cfg $DARKNET/yolov3.weights $DARKNET/data/dog.jpg
