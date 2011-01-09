%.o: %.cpp
	g++ -c $< 
network: main.o bpnn.o
	g++ main.o bpnn.o -o networkTest
clean: 
	rm *.o
