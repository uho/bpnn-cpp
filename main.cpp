#include <iostream>
#include "bpnn.h"
using namespace std;

int main(int argc, char** argv)
{
  //bpnn network(2,2,1,-.1,.1);
  bpnn network("or.NN");
  int numPatterns=4;
  double* inputs = new double[2*numPatterns];
  inputs[0] = 0;
  inputs[1] = 0;
  inputs[2] = 1;
  inputs[3] = 0;
  inputs[4] = 0;
  inputs[5] = 1;
  inputs[6] = 1;
  inputs[7] = 1;
  double* targets = new double[1*numPatterns];
  targets[0] = 0;
  targets[1] = 0;
  targets[2] = 0;
  targets[3] = 1;
  int iterations = 1000;
  double learningRate = .5;
  double momentum = .1;

  /*
  cout << "===================Training===============" << endl;
  network.train(numPatterns, inputs, targets, iterations, learningRate, momentum);
  */

  cout << "====================Testing===============" << endl;
  double* out = network.update(inputs);
  cout << out[0] << endl;
  out = network.update(inputs+2);
  cout << out[0] << endl;
  out = network.update(inputs+4);
  cout << out[0] << endl;
  out = network.update(inputs+6);
  cout << out[0] << endl;

  network.save("or.NN");

  return 0.0;
}
