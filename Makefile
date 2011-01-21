%.o: %.cpp
	g++ -g -c $< 

all: and sin

and: and.o bpnn.o
	g++ -g and.o bpnn.o -o and

sin: sin.o bpnn.o
	g++ sin.o bpnn.o -o sin

clean: 
	rm *.o
