#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>

// The back-propagating neural network class
class bpnn
{
  int ni; // Number of input nodes
  int nh; // Number of hiddn nodes
  int no; // Number of output nodes

  // Note: It would probably be better to implement these as blocked linked lists so that the 
  // memory is not contiguous.  Especially for the matricies.  Also, then there would be no 
  // need to worry about segfaults
  double* ai; // input node activation
  double* ah; // hidden node activation
  double* ao; // output node activation

  // Note: These are actually matricies, to access the ith row jth column, you would do
  // wi[i*numcols+j] and for all these matricies numcols=nh
  double* wi; // input weight matrix
  double* wo; // hidden weight matrix

  double* ci; // input weight change matrix
  double* co; // hidden weight change matrix

  public:
    bpnn(int, int, int, double, double);
    ~bpnn();
    double* update(double*);
    double dsigmoid(double);
    double sigmoid(double);
    void train(int, double*, double*, int, double, double);
    double backPropagate(double*, double, double);
};
