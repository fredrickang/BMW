DEBUG=0

CFLAGS=-ldl -L/usr/local/cuda/lib64 -lcudart 

ifeq ($(DEBUG), 1)
CFLAGS+= -DDEBUG
endif

all: app victim libcuhooklib.so 

app: ./src/app/app.cu
	nvcc -o app ./src/app/app.cu -lrt -cudart shared

victim: ./src/app/victim.cu
	nvcc -o victim ./src/app/victim.cu -lrt -cudart shared

libcuhooklib.so: ./src/hooklib/cuhooklib.cpp
	g++ -I/usr/local/cuda/include -fPIC -shared -o libcuhooklib.so ./src/hooklib/cuhooklib.cpp $(CFLAGS)

clean: 
	rm -f app 
	rm -f victim 
	rm -f libcuhooklib.so

