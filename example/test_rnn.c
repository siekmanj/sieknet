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

int assert_equal(float *v1, float *v2, size_t len){
  int equal = 1;
  for(int i = 0; i < len; i++){
    float diff = v1[i] - v2[i];
    if(diff < 0) diff*=-1;
    if(diff > 0.0001){
      equal = 0;
      //printf("%f is not %f (idx %d)\n", v1[i], v2[i], i);
    }else{
      //printf("%f is %f (idx %d)\n", v1[i], v2[i], i);
    }
  }
  return equal;
}

#ifndef SIEKNET_USE_GPU

float *retrieve_array(float *arr, size_t len){
  return arr;
}
void assign_array(float *arr, float *dest, size_t len){
  for(int i = 0; i < len; i++)
    dest[i] = arr[i];
}
void dispose_array(float *arr){}

#else

float *retrieve_array(cl_mem arr, size_t len){
  float *tmp = (float*)malloc(sizeof(float)*len);
  memset(tmp, '\0', len*sizeof(float)); 
  check_error(clEnqueueReadBuffer(get_opencl_queue0(), arr, 1, 0, sizeof(float) * len, tmp, 0, NULL, NULL), "error reading from gpu (retrieve_array)");
  check_error(clFinish(get_opencl_queue0()), "waiting on queue to finish");
  ALLOCS++;
  return tmp;
}
void assign_array(float *arr, cl_mem dest, size_t len){
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), dest, 1, 0, sizeof(float) * len, arr, 0, NULL, NULL), "enqueuing cost gradient");
  check_error(clFinish(get_opencl_queue0()), "waiting on queue to finish");
}
void dispose_array(float *arr){
  FREES++;
  free(arr);
}
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
  printf("\n******** TESTING RNN FUNCTIONALITY ********\n\n");
#ifndef SIEKNET_USE_GPU
  {
    float x[] = {0.33, 1.00, 0.00, 0.50, 0.25, 0.90};
    float y[] = {0.00, 1.00, 0.00, 0.00, 0.00, 0.00};

    RNN n = create_rnn(6, 50, 6);
    n.seq_len = 3;
    int correct = 1;
    float epsilon = 0.001;
    float threshold = 0.005;
    float norm = 0;
    for(int i = 0; i < n.num_params; i++){
      rnn_wipe(&n);
      float c = 0;
      for(int t = 0; t < n.seq_len; t++){
        rnn_forward(&n, x);
        c += rnn_cost(&n, y);
      }
      rnn_backward(&n);

      float p_grad = n.param_grad[i];

      float c1 = 0;
      n.params[i] += epsilon;
      for(int t = 0; t < n.seq_len; t++){
        rnn_forward(&n, x);
        c1 += rnn_cost(&n, y);
      }
      rnn_backward(&n);
      memset(n.param_grad, '\0', n.num_params*sizeof(float));

      float c2 = 0;
      n.params[i] -= 2*epsilon;
      for(int t = 0; t < n.seq_len; t++){
        rnn_forward(&n, x);
        c2 += rnn_cost(&n, y);
      }
      rnn_backward(&n);
      memset(n.param_grad, '\0', n.num_params*sizeof(float));

      float diff = (p_grad - ((c1 - c2)/(2*epsilon)))/n.seq_len;
      if(diff < 0) diff *=-1;
      if(diff > threshold){ // a fairly generous threshold
        printf("  | (param %d: difference between numerical and actual gradient: %f - %f = %f)\n", i, ((c1 - c2)/(2*epsilon)), p_grad, diff);
        correct = 0;
      }
      memset(n.param_grad, '\0', n.num_params*sizeof(float));
      norm += (diff * diff);
    }
    if(correct)
      printf("  | TEST PASSED: numerical gradient matched calculated gradient (norm %f)\n", sqrt(norm));
    else
      printf("X | TEST FAILED: numerical gradient didn't match calculated gradient (norm %f)\n", sqrt(norm));
  }
#endif
  {
    sleep(1);
    float *tmp, *cmp;
    float x1[] = {0.34, 1.00, 0.00, 0.25, 0.89};
    float x2[] = {0.99, 0.00, 1.00, 0.59, 0.89};

    RNN n = create_rnn(5, 4, 5);
    rnn_wipe(&n);
    
    rnn_forward(&n, x1);
    tmp = retrieve_array(n.layers[0].output[0], n.layers[0].size);
    rnn_forward(&n, x2);
    cmp = retrieve_array(n.layers[0].loutput, n.layers[0].size);

    if(!assert_equal(tmp, cmp, n.layers[0].size)){
      printf("X | TEST FAILED: RNN recurrent input did not match last timestep's output (without incrementing n.t)\n");
    }else{
      printf("  | TEST PASSED: RNN recurrent input matched last timestep's output (without incrementing n.t)\n");
    }

    rnn_cost(&n, x1);
    rnn_backward(&n);
  }
}
