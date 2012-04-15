#include <iostream>
#include "bpnn.h"
using namespace std;

int main(int argc, char** argv)
{
  int validInp = 1;
  int training = 0;
  int run = 0;

  int ni;
  int nh;
  int no;
  
  if(argc!=4 && argc!=7)
    validInp = 0;
  if(validInp == 1 && argv[1][0] == 'r')
    training = 1;
  else if(validInp == 1 && argv[1][0] == 'e')
    training = 0;
  else if(validInp == 1 && argv[1][0] == 'u')
    run = 1;
  else
    validInp = 0;

  if(validInp == 1 && (training == 1 || run == 1) && argc==7)
  {
    ni = atoi(argv[4]);
    nh = atoi(argv[5]);
    no = atoi(argv[6]);
  }
  else if(training==1)
    validInp=0;

  if(validInp == 0)
  {
    cout << "Usage: and [reu] filenameData filenameNetwork [ni nh no]" << endl << "r means"
    " training, e means testing, and u means run. If training is chosen, then ni, nh, and no must be specified." << endl;
    return 1;
  }

  if(training == 0 && run != 1)
  {
    cout << "====================Testing===============" << endl;
    // Load network
    bpnn network(argv[3]);
    // Test on data from file
    network.test(argv[2]);
  }
  if(training == 1 && run != 1)
  {
    cout << "===================Training===============" << endl;
    // Determine ni, nh, and no from data file
    
    // Initialize network
    bpnn network(ni, nh, no, -.1, .1);
    // Train on data from file
    network.train(argv[2]);
    // Save to file
    network.save(argv[3]);
  }
  if(run == 1)
  {
    // Load network
    bpnn network(argv[3]);
    // Run the network on data from file
    network.run(argv[2]);
  }

  return 0;

}
