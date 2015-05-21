CC=gcc
FLAGS=-Wall

HEADERS = common.h graph.h
OBJECTS = common.o graph.o gpm-seq-naive.o

default: gpm-seq-naive

%.o: %.c $(HEADERS)
	$(CC) $(FLAGS) -c $< -o $@


gpm-seq-naive: $(OBJECTS)
	$(CC) $(OBJECTS) $(FLAGS) -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f gpm-sec

