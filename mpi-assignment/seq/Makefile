CC=gcc
FLAGS=-Wall

HEADERS = common.h graph.h
OBJECTS = common.o graph.o gpm-sec-naive.o

default: gpm-sec-naive

%.o: %.c $(HEADERS)
	$(CC) $(FLAGS) -c $< -o $@


gpm-sec-naive: $(OBJECTS)
	$(CC) $(OBJECTS) $(FLAGS) -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f gpm-sec

