#include "optimizer.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef SIEKNET_USE_GPU
static void cpu_sgd_step(SGD o){
  for(int i = 0; i < o.num_params; i++){
    o.weights[i] += o.learning_rate * o.gradient[i];
    o.gradient[i] = 0.0;
  }
}

static void cpu_momentum_step(Momentum o){
  for(int i = 0; i < o.num_params; i++){
    o.z[i] = o.beta * o.z[i] + o.alpha * o.gradient[i];
    o.weights[i] += o.z[i];
    o.gradient[i] = 0.0;
  }
}

SGD cpu_init_SGD(float *weights, float *gradient, size_t num_params){
  SGD o;
  o.weights = weights;
  o.gradient = gradient;
  o.num_params = num_params;
  o.learning_rate = 0.05;
  o.step = cpu_sgd_step;
  return o;
}

Momentum cpu_init_Momentum(float *weights, float *gradient, size_t num_params){
  Momentum o;
  o.weights = weights;
  o.gradient = gradient;

  o.z = (float*)malloc(num_params*sizeof(float));
  memset(o.z, '\0', num_params*sizeof(float));

  o.num_params = num_params;
  o.alpha = 0.001;
  o.beta = 0.99;
  o.step = cpu_momentum_step;
  return o;
}

#endif

#ifdef SIEKNET_USE_GPU
static cl_kernel sgd_step_kernel, momentum_step_kernel;

static int ARE_KERNELS_INITIALIZED = 0;
void optimizer_gpu_setup(){
  char *kernels[] = {"src/optimizer.cl"};

  char *src = get_kernel_source(kernels, 1);

  cl_program prog = build_program(src);
  free(src);

  int err;
  sgd_step_kernel = clCreateKernel(prog, "sgd_step_kernel", &err);
  check_error(err, "couldn't make sgd kernel");

  momentum_step_kernel = clCreateKernel(prog, "momentum_step_kernel", &err);
  check_error(err, "couldn't make momentum kernel");

  ARE_KERNELS_INITIALIZED = 1;
}

static void gpu_sgd_step(SGD o){
  check_error(clSetKernelArg(sgd_step_kernel, 0, sizeof(cl_mem), &o.weights), "setting sgd step kernel arg 0");
  check_error(clSetKernelArg(sgd_step_kernel, 1, sizeof(cl_mem), &o.gradient), "setting sgd step kernel arg 0");
  check_error(clSetKernelArg(sgd_step_kernel, 2, sizeof(float), &o.learning_rate), "setting sgd step kernel arg 0");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), sgd_step_kernel, 1, NULL, &o.num_params, NULL, 0, NULL, NULL), "couldn't enqueue param update kernel");
}

static void gpu_momentum_step(Momentum o){
  check_error(clSetKernelArg(momentum_step_kernel, 0, sizeof(cl_mem), &o.weights), "setting mom step kernel arg 0");
  check_error(clSetKernelArg(momentum_step_kernel, 1, sizeof(cl_mem), &o.gradient), "setting mom step kernel arg 1");
  check_error(clSetKernelArg(momentum_step_kernel, 2, sizeof(cl_mem), &o.z), "setting mom step kernel arg 1");
  check_error(clSetKernelArg(momentum_step_kernel, 3, sizeof(float), &o.alpha), "setting mom kernel arg 2");
  check_error(clSetKernelArg(momentum_step_kernel, 4, sizeof(float), &o.beta), "setting mom step kernel arg 3");
  check_error(clEnqueueNDRangeKernel(get_opencl_queue0(), momentum_step_kernel, 1, NULL, &o.num_params, NULL, 0, NULL, NULL), "couldn't enqueue param update kernel");
}

SGD gpu_init_SGD(cl_mem weights, cl_mem gradient, size_t num_params){
  if(!ARE_KERNELS_INITIALIZED)
    optimizer_gpu_setup();
  SGD o;
  o.weights = weights;
  o.gradient = gradient;
  o.num_params = num_params;
  o.learning_rate = 0.05;
  o.step = gpu_sgd_step;
  return o;
}

Momentum gpu_init_Momentum(cl_mem weights, cl_mem gradient, size_t num_params){
  if(!ARE_KERNELS_INITIALIZED)
    optimizer_gpu_setup();
  Momentum o;
  o.weights = weights;
  o.gradient = gradient;

  int err;
  float *zeros = (float*)malloc(sizeof(float)*num_params);
  memset(zeros, '\0', num_params*sizeof(float));

  o.z = clCreateBuffer(get_opencl_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_params, zeros, &err);
  check_error(err, "could not create momentum buffer");
  free(zeros);

  o.num_params = num_params;
  o.alpha = 0.001;
  o.beta = 0.99;
  o.step = gpu_momentum_step;
  return o;
}
#endif
