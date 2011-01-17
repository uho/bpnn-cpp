#include "bpnn.h"

// The destructor, it frees all the memory that is allocated by the constructor
bpnn::~bpnn()
{
  delete [] ai; ai=NULL;
  delete [] ah; ah=NULL;
  delete [] ao; ao=NULL;

}
// The constructor, it loads a neural network from a file
bpnn::bpnn(char* filename)
{
  ifstream fin;
  fin.open(filename);
  fin >> ni;
  fin >> nh;
  fin >> no;

  // Initialize Activations
  ai = new double[ni];
  ah = new double[nh];
  ao = new double[no];
  // Set nodes to 1
  for(int i=0; i<ni; i++)
    ai[i] = 1.0;
  for(int i=0; i<nh; i++)
    ah[i] = 1.0;
  for(int i=0; i<no; i++)
    ao[i] = 1.0;

  // Initialize Weights
  wi.resize(ni*nh);
  wo.resize(nh*no);

  // Read Weights from file
  for(int i=0; i<ni*nh; i++)
    fin >> wi[i];
  for(int i=0; i<nh*no; i++)
    fin >> wo[i];
  fin.close();

  // Initialize Change in Weights to 0
  ci.resize(ni*nh,0);
  co.resize(nh*no,0);
}
// The constructor, it creates a nerural network with the given numbers of nodes, and with 
// weights initialized to random numbers between min and max
bpnn::bpnn(int niin, int nhin, int noin, double min, double max)
{
  // Seed the random number generator (note: for large neural networks the standard c++ random 
  // number generator might not be random "enough")
  srand(time(0));
  ni = niin+1; // the +1 is for the bias node (what is the bias node?) --> the bias node is always set to 1, I'm still not sure what it does exactly
  nh = nhin;
  no = noin;

  // Initialize Activations
  ai = new double[ni];
  ah = new double[nh];
  ao = new double[no];
  // Set nodes to 1
  for(int i=0; i<ni; i++)
    ai[i] = 1.0;
  for(int i=0; i<nh; i++)
    ah[i] = 1.0;
  for(int i=0; i<no; i++)
    ao[i] = 1.0;

  // Initialize Weights
  wi.resize(ni*nh);
  wo.resize(nh*no);

  // Set them to random values within the range [min,max]
  for(int i=0; i<ni*nh; i++)
    wi[i] = min + (double)rand() / (double)RAND_MAX * (max - min);
  for(int i=0; i<nh*no; i++)
    wo[i] = min + (double)rand() / (double)RAND_MAX * (max - min);

  // Initialize Change in Weights to 0
  ci.resize(ni*nh,0);
  co.resize(nh*no,0);
}

double* bpnn::update(double* inputs)
{
  // Initialize input activations
  for(int i=0; i<ni-1; i++)
    ai[i] = inputs[i];

  // Calculate hidden activations
  for(int j=0; j<nh; j++)
  {
    double sum = 0.0;
    for(int i=0; i<ni; i++)
      sum += ai[i] * wi[i*nh+j];
    ah[j] = sigmoid(sum);
  }

  // Output activations
  for(int k=0; k<no; k++)
  {
    double sum = 0.0;
    for(int j=0; j<nh; j++)
      sum += ah[j]*wo[j*no+k];
    ao[k] = sigmoid(sum);
  }
  return ao;
}

// targets is the list of expected output node activations
// note: don't forget to update before backPropagating
double bpnn::backPropagate(double* targets, double learningRate, double momentum)
{
/* implement when the double pointers are replaced by linked lists or some other equivalent
   data structure
    if len(targets) != self.no:
        raise ValueError('wrong number of target values')
*/

  // calculate error terms for output
  double* outputDeltas = new double[no];
  for(int i=0; i<no; i++)
    outputDeltas[i] = 0.0;

  for(int k=0; k<no; k++)
    outputDeltas[k] = dsigmoid(ao[k]) * (targets[k] - ao[k]);

  // calculate error terms for hidden
  double* hiddenDeltas = new double[nh];
  for(int i=0; i<nh; i++)
    hiddenDeltas[i] = 0.0;

  for(int j=0; j<nh; j++)
  {
    double error = 0.0;
    for(int k=0; k<no; k++)
      error += outputDeltas[k]*wo[j*no+k];
    hiddenDeltas[j] = dsigmoid(ah[j])*error;
  }

  // update output weights
  for(int j=0; j<nh; j++)
  {
    for(int k=0; k<no; k++)
    {
      double change = outputDeltas[k]*ah[j];
      wo[j*no+k] += learningRate*change + momentum*co[j*no+k];
      co[j*no+k] = change;
    }
  }

  // update input weights
  for(int i=0; i<ni; i++)
  {
    for(int j=0; j<nh; j++)
    {
      double change = hiddenDeltas[j]*ai[i];
      wi[i*nh+j] = wi[i*nh+j] + learningRate*change + momentum*ci[i*nh+j];
      ci[i*nh+j] = change;
    }
  }

  // calculate error
  double error = 0.0;
  for(int k=0; k<no; k++)
    error = error + 0.5*(targets[k]-ao[k])*(targets[k]-ao[k]);

  return error;
}

void bpnn::train(int numPatterns, double* inputs, double* targets, int iterations, double learningRate, double momentum)
{
  for(int i=0; i<iterations; i++)
  {
    double error = 0.0;
    for(int j=0; j<numPatterns; j++)
    {
      update(inputs+j*(ni-1));
      error += backPropagate(targets+j*no, learningRate, momentum);
    }
    std::cout << "error is " << error << std::endl;
  }
}

double bpnn::sigmoid(double x)
{
  return tanh(x);
}

// note: I ran into some problems where the derivative would flatten out if x was too large
// or too small.  Simply multiplying the derivative by a number seemed to fix the problem, and
// it didn't matter what the number was, all that mattered was that the derivative not flatten
// out. I think this is called neuron saturation or something, I read about it in some article
// online.
double bpnn::dsigmoid(double x)
{
  return 1.0 - x*x;
}

void bpnn::save(char* filename)
{
  ofstream fout;
  fout.open(filename);
  fout << ni << endl;
  fout << nh << endl;
  fout << no << endl;

  // Write weights to file
  for(int i=0; i<ni*nh; i++)
    fout << wi[i] << " ";
  fout << endl;
  for(int i=0; i<nh*no; i++)
    fout << wo[i] << " ";
  fout << endl;
  fout.close();
}
