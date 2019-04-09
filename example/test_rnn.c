#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <rnn.h>
#include <unistd.h>
#include <conf.h>

#ifdef SIEKNET_USE_GPU
#include <opencl_utils.h>
#endif

int main(){
  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_  __/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/     \n");
  printf("																					 \n");
  printf("preparing to run rnn unit test suite...\n");
  sleep(1);

  srand(1);

  /* LSTM tests */
  {
    printf("\n******** TESTING LSTM FUNCTIONALITY ********\n\n");
    sleep(1);
    float x1[] = {0.34, 1.00, 0.00, 0.25, 0.89};
    float x2[] = {0.99, 0.00, 1.00, 0.59, 0.89};
    float x3[] = {0.13, 0.66, 0.72, 0.01, 0.89};

    RNN n = create_rnn(5, 3, 4, 5);
    
    rnn_forward(&n, x1);
    PRINTLIST(n.output, n.output_dimension);
    rnn_cost(&n, x1);
    rnn_backward(&n);
  }
}
