/* Jonah Siekmann
 * 1/20/2019
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <mlp.h>
#include <optimizer.h>

#define createonehot(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
//#define printlist(name, len) printf("printing %s: {", #name); for(int xyz = 0; xyz < len; xyz++){printf("%f", name[xyz]); if(xyz < len-1) printf(", "); else printf("};\n");}

float uniform(float minimum, float maximum){
  float center = minimum + (maximum - minimum)/2;
  float max_mag = maximum - center;
  if(rand()&1)
    return center + ((((float)rand())/RAND_MAX)) * max_mag;
  else
    return center - ((((float)rand())/RAND_MAX)) * max_mag;
}

int assert_equal(float *v1, float *v2, size_t len){
  for(int i = 0; i < len; i++){
    float diff = v1[i] - v2[i];
    if(diff < 0) diff*=-1;
    if(diff > 0.0001){
      //printf("%f is not %f\n", v1[i], v2[i]);
      return 0;
    }
  }
  return 1;
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
  return tmp;
}
void assign_array(float *arr, cl_mem dest, size_t len){
  check_error(clEnqueueWriteBuffer(get_opencl_queue0(), dest, 1, 0, sizeof(float) * len, arr, 0, NULL, NULL), "enqueuing cost gradient");
}
void dispose_array(float *arr){
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
  printf("preparing to run mlp unit test suite...\n");
  sleep(1);

  srand(1);
  printf("\n******** TESTING MLP FUNCTIONALITY ********\n\n");
  {
    sleep(1);
    MLP n = create_mlp(4, 6, 7, 5);

    int success = 1;
    for(int i = 0; i < n.depth; i++){
      if(i < n.depth-1 && n.layers[i].logistic != sigmoid){
        printf("X | TEST FAILED: Default hidden layer logistic function was not sigmoid!\n");
        success = 0;
      }else if(i == n.depth-1 && n.layers[i].logistic != softmax){
        printf("X | TEST FAILED: Default output layer logistic function was not softmax!\n");
        success = 0;
      }
    }
    if(success)
      printf("  | TEST PASSED: Default logistic functions were as expected.\n");

    float x[]  = {0.50, 1.2, 0.01, 0.75};
    float y[]  = {0.25, 0.09, 0.0, 1.0, 0.75};
    float l0[] = {0.4103, 0.5631, 0.4348, 0.5297, 0.3943, 0.4167};
    float l1[] = {0.3545, 0.5293, 0.5700, 0.6833, 0.3192, 0.4952, 0.4268};
    float l2[] = {0.3459, 0.1410, 0.0896, 0.2343, 0.1892};
    
    float c[]  = {0.4238, 0.4210, 1.0984, -4.2683, -3.6548};
    float g0[] = {-0.0020, -0.0047, -0.0011, -0.0011};
    float g1[] = {0.0319, -0.0119, -0.0122, -0.0136, -0.0118, 0.0174};
    float g2[] = {-0.3366, -0.0155, 0.0865, -0.4037, 0.0830, 0.1082, 0.0088};

    float p[]  = {0.3583, -0.3339, -0.3887, 0.1429, -0.1184, -0.2036, -0.1555, 0.4060, 0.2711, 0.0604, 0.0070, -0.0585, 0.0668, -0.0553, -0.4259, 0.2187, 0.2612, -0.2719, 0.2105, 0.1247, -0.2246, -0.1707, -0.1208, 0.3444, 0.0297, 0.2243, 0.0820, -0.3796, 0.0274, -0.1952, -0.0901, -0.3410, -0.1008, -0.1418, -0.1937, -0.2009, -0.1654, -0.3518, 0.1075, 0.2419, 0.2600, 0.1663, 0.3134, -0.0865, -0.1324, 0.3615, 0.2484, 0.1661, 0.1506, -0.2586, 0.1824, 0.3592, -0.0558, 0.2423, 0.2342, 0.2971, 0.1690, -0.0709, -0.2103, -0.0641, -0.0390, -0.1873, -0.3722, -0.2587, -0.2834, 0.1112, -0.2209, -0.0576, -0.0474, 0.0620, -0.0282, -0.0199, -0.0666, 0.3015, -0.2482, -0.2417, -0.0353, -0.1966, -0.0264, -0.1810, 0.2249, 0.0204, 0.3922, 0.3491, 0.3914, 0.3415, 0.0016, -0.2329, -0.0640, 0.3581, -0.1408, 0.2273, -0.2696, -0.2082, -0.1194, -0.2263, -0.2933, 0.0139, -0.3268, -0.3425, 0.3842, -0.3543, 0.2616, 0.0643, 0.3487, 0.2548, -0.2469, 0.2748, -0.1290, -0.0291, -0.0877, -0.2003, 0.1098, -0.2821, 0.1849, 0.3704, 0.1319, -0.1704, 0.1353};
    float pg[] = {0.007700, 0.003900, 0.009300, 0.000100, 0.005800, -0.002900, -0.001500, -0.003500, -0.000000, -0.002200, -0.003000, -0.001500, -0.003600, -0.000000, -0.002200, -0.003400, -0.001700, -0.004100, -0.000000, -0.002500, -0.002800, -0.001400, -0.003400, -0.000000, -0.002100, 0.004200, 0.002100, 0.005100, 0.000000, 0.003200, -0.077000, -0.031600, -0.043400, -0.033500, -0.040800, -0.030400, -0.032100, -0.003900, -0.001600, -0.002200, -0.001700, -0.002000, -0.001500, -0.001600, 0.021200, 0.008700, 0.011900, 0.009200, 0.011200, 0.008400, 0.008800, -0.087400, -0.035900, -0.049200, -0.038000, -0.046300, -0.034500, -0.036400, 0.018000, 0.007400, 0.010200, 0.007800, 0.009600, 0.007100, 0.007500, 0.027100, 0.011100, 0.015200, 0.011800, 0.014300, 0.010700, 0.011300, 0.002100, 0.000900, 0.001200, 0.000900, 0.001100, 0.000800, 0.000900, 0.095900, 0.034000, 0.050800, 0.054700, 0.065500, 0.030600, 0.047500, 0.040900, 0.051000, 0.018100, 0.027000, 0.029100, 0.034800, 0.016300, 0.025200, 0.021800, 0.089600, 0.031800, 0.047400, 0.051100, 0.061200, 0.028600, 0.044400, 0.038200, -0.765700, -0.271500, -0.405300, -0.436500, -0.523200, -0.244500, -0.379200, -0.326800, -0.560800, -0.198800, -0.296800, -0.319600, -0.383200, -0.179000, -0.277700, -0.239300};

    assign_array(p, n.params, n.num_params);

    mlp_forward(&n, x);
    float *tmp = retrieve_array(n.layers[0].output, n.layers[0].size);
    if(!assert_equal(tmp, l0, n.layers[0].size)){
      printf("X | TEST FAILED: First hidden layer's output was incorrect.\n");
    }else{
      printf("  | TEST PASSED: First hidden layer's output was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].output, n.layers[1].size);
    if(!assert_equal(tmp, l1, n.layers[1].size)){
      printf("X | TEST FAILED: Second hidden layer's output was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Second hidden layer's output was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[2].output, n.layers[2].size);
    if(!assert_equal(tmp, l2, n.layers[2].size)){
      printf("X | TEST FAILED: Third hidden layer's output was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Third hidden layer's output was as expected.\n");
    }
    dispose_array(tmp);

    printf("\n");
    float c0 = mlp_cost(&n, y) - 3.744455;
    if(c0 < 0) c0*=-1;
    if(c0 > 0.0001){
      printf("X | TEST FAILED: Cost scalar was not calculated correctly.\n");
    }else{
      printf("  | TEST PASSED: Cost scalar was as expected.\n");
    }

    tmp = retrieve_array(n.cost_gradient, n.output_dimension);
    if(!assert_equal(tmp, c, n.output_dimension)){
      printf("X | TEST FAILED: Cost gradient was not calculated correcly.\n");
    }else{
      printf("  | TEST PASSED: Cost gradient was as expected.\n");
    }
    dispose_array(tmp);

    printf("\n");
    mlp_backward(&n);

    tmp = retrieve_array(n.layers[0].input_gradient, n.layers[0].input_dimension);
    if(!assert_equal(tmp, g0, n.layers[0].input_dimension)){
      printf("X | TEST FAILED: First hidden layer's input gradient was incorrect.\n");
    }else{
      printf("  | TEST PASSED: First hidden layer's input gradient was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[1].input_gradient, n.layers[1].input_dimension);
    if(!assert_equal(tmp, g1, n.layers[1].input_dimension)){
      printf("X | TEST FAILED: Second hidden layer's input gradient was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Second hidden layer's input gradient was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.layers[2].input_gradient, n.layers[2].input_dimension);
    if(!assert_equal(tmp, g2, n.layers[2].input_dimension)){
      printf("X | TEST FAILED: Third hidden layer's input gradient was incorrect.\n");
    }else{
      printf("  | TEST PASSED: Third hidden layer's input gradient was as expected.\n");
    }
    dispose_array(tmp);

    tmp = retrieve_array(n.param_grad, n.num_params);
    if(!assert_equal(tmp, pg, n.num_params)){
      printf("X | TEST FAILED: Network parameter gradients were incorrect.\n");
    }else{
      printf("  | TEST PASSED: Network parameter gradients were as expected.\n");
    }
    dispose_array(tmp);

    save_mlp(&n, "./model/test.mlp");

    MLP m = load_mlp("./model/test.mlp");

    float *tmp1 = retrieve_array(n.params, n.num_params);
    float *tmp2 = retrieve_array(m.params, m.num_params);
    if(!assert_equal(tmp1, tmp2, n.num_params)){
      printf("X | TEST FAILED: MLP saved to disk did not match once loaded back into memory.\n");
    }else{
      printf("  | TEST PASSED: MLP saved to disk was identical to original.\n");
    }
    dispose_array(tmp1);
    dispose_array(tmp2);

    dealloc_mlp(&m);

    dealloc_mlp(&n);
    sleep(1);
  }
  printf("\n");
  {

    float x[] = {0.05, 0.01, 0.10};
    float y[] = {0.00, 0.00, 1.00};
    //           b1     w1    w2    b1    w3    w4    b2    w5    w6    b2    w7    w8
    //float p[] = {0.35, 0.15, 0.20, 0.35, 0.25, 0.30, 0.60, 0.40, 0.45, 0.60, 0.50, 0.55};

    MLP n = create_mlp(3, 5, 5, 3);
    //n.cost_fn = quadratic;
    n.layers[n.depth-1].logistic = sigmoid;
    //assign_array(p, n.params, n.num_params);
    
    int correct = 1;
    for(int i = 0; i < n.num_params; i++){
      mlp_forward(&n, x);
      float c = mlp_cost(&n, y);
      mlp_backward(&n);

      int p_idx = i;
      float epsilon = 0.01;
      float p_grad = n.param_grad[p_idx];

      n.params[p_idx] += epsilon;
      mlp_forward(&n, x);
      float c1 = mlp_cost(&n, y);

      n.params[p_idx] -= 2*epsilon;
      mlp_forward(&n, x);
      float c2 = mlp_cost(&n, y);
      
      float diff = p_grad - ((c1 - c2)/(2*epsilon));
      if(diff < 0) diff *=-1;
      if(diff > 0.005){ // a fairly generous threshold
        printf("  | (difference between numerical and actual gradient: %f - %f = %f)\n", ((c1 - c2)/(2*epsilon)), p_grad, diff);
        correct = 0;
      }

      memset(n.param_grad, '\0', n.num_params*sizeof(float));
    }
    if(correct)
      printf("  | TEST PASSED: Numerical gradient checking was successful.\n");
    else
      printf("X | TEST FAILED: Numerical gradient checking showed inconsistency - check gradient math.\n");

    dealloc_mlp(&n);

  }
}
