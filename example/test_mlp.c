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
#define printlist(name, len) printf("printing %s: {", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("};\n");}

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
  /* MLP tests */
  {
    printf("\n******** TESTING MLP FUNCTIONALITY ********\n\n");
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
    float c[]  = {-0.4238, -0.4210, -1.0984, 4.2683, 3.6548};
    float g0[] = {0.0020, 0.0047, 0.0011, 0.0011};
    float g1[] = {-0.0319, 0.0119, 0.0122, 0.0136, 0.0118, -0.0174};
    float g2[] = {0.3366, 0.0155, -0.0865, 0.4037, -0.0830, -0.1082, -0.0088};
    float p[]  = {0.3583, -0.3339, -0.3887, 0.1429, -0.1184, -0.2036, -0.1555, 0.4060, 0.2711, 0.0604, 0.0070, -0.0585, 0.0668, -0.0553, -0.4259, 0.2187, 0.2612, -0.2719, 0.2105, 0.1247, -0.2246, -0.1707, -0.1208, 0.3444, 0.0297, 0.2243, 0.0820, -0.3796, 0.0274, -0.1952, -0.0901, -0.3410, -0.1008, -0.1418, -0.1937, -0.2009, -0.1654, -0.3518, 0.1075, 0.2419, 0.2600, 0.1663, 0.3134, -0.0865, -0.1324, 0.3615, 0.2484, 0.1661, 0.1506, -0.2586, 0.1824, 0.3592, -0.0558, 0.2423, 0.2342, 0.2971, 0.1690, -0.0709, -0.2103, -0.0641, -0.0390, -0.1873, -0.3722, -0.2587, -0.2834, 0.1112, -0.2209, -0.0576, -0.0474, 0.0620, -0.0282, -0.0199, -0.0666, 0.3015, -0.2482, -0.2417, -0.0353, -0.1966, -0.0264, -0.1810, 0.2249, 0.0204, 0.3922, 0.3491, 0.3914, 0.3415, 0.0016, -0.2329, -0.0640, 0.3581, -0.1408, 0.2273, -0.2696, -0.2082, -0.1194, -0.2263, -0.2933, 0.0139, -0.3268, -0.3425, 0.3842, -0.3543, 0.2616, 0.0643, 0.3487, 0.2548, -0.2469, 0.2748, -0.1290, -0.0291, -0.0877, -0.2003, 0.1098, -0.2821, 0.1849, 0.3704, 0.1319, -0.1704, 0.1353};
    float pg[] = {-0.0077, -0.0039, -0.0093, -0.0001, -0.0058, 0.0029, 0.0015, 0.0035, 0.0000, 0.0022, 0.0030, 0.0015, 0.0036, 0.0000, 0.0022, 0.0034, 0.0017, 0.0041, 0.0000, 0.0025, 0.0028, 0.0014, 0.0034, 0.0000, 0.0021, -0.0042, -0.0021, -0.0051, -0.0000, -0.0032, 0.0770, 0.0316, 0.0434, 0.0335, 0.0408, 0.0304, 0.0321, 0.0039, 0.0016, 0.0022, 0.0017, 0.0020, 0.0015, 0.0016, -0.0212, -0.0087, -0.0119, -0.0092, -0.0112, -0.0084, -0.0088, 0.0874, 0.0359, 0.0492, 0.0380, 0.0463, 0.0345, 0.0364, -0.0180, -0.0074, -0.0102, -0.0078, -0.0096, -0.0071, -0.0075, -0.0271, -0.0111, -0.0152, -0.0118, -0.0143, -0.0107, -0.0113, -0.0021, -0.0009, -0.0012, -0.0009, -0.0011, -0.0008, -0.0009, -0.0959, -0.0340, -0.0508, -0.0547, -0.0655, -0.0306, -0.0475, -0.0409, -0.0510, -0.0181, -0.0270, -0.0291, -0.0348, -0.0163, -0.0252, -0.0218, -0.0896, -0.0318, -0.0474, -0.0511, -0.0612, -0.0286, -0.0444, -0.0382, 0.7657, 0.2715, 0.4053, 0.4365, 0.5232, 0.2445, 0.3792, 0.3268, 0.5608, 0.1988, 0.2968, 0.3196, 0.3832, 0.1790, 0.2777, 0.2393};

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

    dealloc_mlp(&n);
    dealloc_mlp(&m);

    sleep(1);
  }
}
