%.o: %.cpp
	g++ -g -c $< 

main: main.o bpnn.o
	g++ -g main.o bpnn.o -o NN

clean: 
	rm *.o
