all: app victim libcuhooklib.so libchooklib.so

app: app.cu
	nvcc -o app app.cu -cudart shared

victim: victim.cu
	nvcc -o victim victim.cu -cudart shared

libcuhooklib.so: cuhooklib.cpp
	g++ -I/usr/local/cuda/include -fPIC -shared -o libcuhooklib.so cuhooklib.cpp -ldl -L/usr/local/cuda/lib64 -lcudart

libchooklib.so: chooklib.c
	gcc -fPIC -shared -o libchooklib.so chooklib.c 

clean: 
	rm -f app 
	rm -f libcuhooklib.so
	rm -f libchooklib.so
