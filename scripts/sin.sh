echo "=====Sin wave example====="
./NN r ./data/sin.dat ./networks/sin.NN 100 4 1
./NN e ./data/sinTest.dat ./networks/sin.NN
