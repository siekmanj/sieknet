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
  {
    printf("\n******** TESTING RNN FUNCTIONALITY ********\n\n");
    sleep(1);
    float *tmp, *cmp;
    float x1[] = {0.34, 1.00, 0.00, 0.25, 0.89};
    float x2[] = {0.99, 0.00, 1.00, 0.59, 0.89};
    float x3[] = {0.13, 0.66, 0.72, 0.01, 0.89};

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

    PRINTLIST(n.output, n.output_dimension);
    
    float c = rnn_cost(&n, x1);
    rnn_backward(&n);
  }
}
