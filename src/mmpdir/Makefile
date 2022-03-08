DEBUG=1

CC=g++
CFLAGS=-g -Wall -std=c++11
OBJS=mmp.o mmp_fn.o
TARGET=mmp

OPTS=-Ofast
CFLAGS+=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
CFLAGS+=-DDEBUG
endif

CFLAGS+=$(OPTS)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $@  $(OBJS)

mmp.o: mmp.hpp mmp_fn.hpp mmp.cpp
mmp_fn.o: mmp_fn.hpp mmp.hpp mmp_fn.cpp

clean:
	rm -f *.o
	rm -f $(TARGET)
