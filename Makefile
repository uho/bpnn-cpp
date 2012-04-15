%.o: %.cpp
	g++ -c $< 

main: main.o bpnn.o
	g++ main.o bpnn.o -o NN

clean: 
	rm *.o
