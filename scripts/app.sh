echo "xavier5" |sudo -S LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=./libcuhooklib.so ./app -period 500 -prio 2
