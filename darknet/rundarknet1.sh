LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=../libcuhooklib.so ./darknet detector test cfg/coco.data cfg/yolov3_p1.cfg yolov3.weights data/dog.jpg
