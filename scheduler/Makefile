DEBUG=1
MMP=1
CC=g++
CFLAGS=-g -Wall -std=c++11
OBJS=scheduler.o scheduler_fn.o
TARGET=scheduler

OPTS=-Ofast
CFLAGS+=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(DEBUG), 1)
OPTS=-O0 -g
CFLAGS+=-DDEBUG
endif

ifeq ($(MMP), 1)
CFLAGS+=-DMMP
endif


CFLAGS+=$(OPTS)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $@  $(OBJS)

scheduler.o: scheduler.h scheduler_fn.h scheduler.c
scheduler_fn.o: scheduler_fn.h scheduler.h scheduler_fn.c

clean:
	rm -f *.o
	rm -f $(TARGET)
