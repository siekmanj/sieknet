#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <conf.h>
#include <logistic.h>
#include <mlp.h>

#ifdef SIEKNET_USE_GPU
#include <CL/cl.h>
#endif

#define create_rnn(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct rnn_layer{
#ifndef SIEKNET_USE_GPU 
  float **z;
  float **output;
  float **input;
  float **input_gradient;

  float *loutput;
#else
  cl_mem *z;
  cl_mem *output;
  cl_mem *input;
  cl_mem *input_gradient;

  cl_mem loutput;
#endif

  Nonlinearity logistic;

  int param_offset;
  size_t input_dimension;
  size_t size;
} RNN_layer;

typedef struct rnn {
#ifndef SIEKNET_USE_GPU
  float *params;
  float *param_grad;
  float **recurrent_gradient;
  float **network_input;
  
  float *cost_gradient;
#else
  cl_mem params;
  cl_mem param_grad;
  cl_mem *recurrent_gradient;
  cl_mem *network_input;

  cl_mem cost_gradient;
  cl_mem output_label;
#endif
  float *output;
  float performance;

  int stateful;
  int guess;

  size_t input_dimension;
  size_t output_dimension;

  size_t num_params;
  size_t seq_len;
  size_t depth;
  size_t t;
  
  RNN_layer *layers;
  MLP_layer output_layer;
  Costfn cost_fn;

} RNN;

RNN rnn_from_arr(const size_t *, const size_t);
RNN load_rnn(const char *);
RNN *copy_rnn(RNN *n);
void save_rnn(RNN *n, const char *);

void rnn_forward(RNN *, const float *);
void rnn_backward(RNN *);
void rnn_abs_backward(RNN *);
float rnn_cost(RNN *, const float *);

void rnn_wipe(RNN *);

void dealloc_rnn(RNN *);

/*
 * In this file, rnn kernels are implemented as static functions to be used in both the gpu and cpu
 * implementation of sieknet.
 *
 * This was a design decision to emphasize re-use of code, and enforce under-the-hood
 * homogeneity across CPU/GPU implementations. Unfortunately, OpenCL does not allow address
 * space changes (i.e., passing a __global pointer to a function that takes a pointer), which
 * necessitated the use of macros to provide an implementation that could be reused on the 
 * GPU as well as the CPU. When compiled with OpenCL, __mem_ro and __mem_rw are defined as
 * either __constant or const __global, and __global. When compiled on the host machine,
 * mem_rw is defined as an empty macro, and __mem_ro is defined as const. This allows the same
 * code to be reused on the CPU.
 */


#define __mem_rw
#define __mem_ro const

/*<<KERNEL START>>*/

#if !defined(__mem_rw)
#define __mem_rw __global
#endif

#if defined(SIEKNET_AMDGPU_READONLY_SPEEDUP) && !defined(__mem_ro)
#define __mem_ro __constant
#endif

#if !defined(SIEKNET_AMDGPU_READONLY_SPEEDUP) && !defined(__mem_ro)
#define __mem_ro const __global
#endif

static void agnostic_rnn_forward_kernel(__mem_ro float *x, 
                                        __mem_ro float *r, 
                                        __mem_rw float *z, 
                                        __mem_ro float *params, 
                                        const int dim,
                                        const int size,
                                        const int layer_param_idx,
                                        const int skiplength,
                                        const int i){
  z[i] = 0.0f;                                             
  const int w_idx = layer_param_idx + (skiplength * i);    
  for(int j = 0; j < dim-size; j++)                  
    z[i] += x[j] * params[w_idx + j + 1];              
  for(int j = 0; j < size; j++)                      
    z[i] += r[j] * params[w_idx + (dim-size) + j + 1]; 
  z[i] += params[w_idx];                                  
}

static void agnostic_rnn_input_gradient_kernel(__mem_ro float *gradient,
                                               __mem_ro float *output,
                                               __mem_ro float *params,
                                               __mem_ro float *future_input_gradient,
                                               __mem_rw float *input_gradient,
                                               const Nonlinearity logistic,
                                               const int use_future_gradient,
                                               const int dim,
                                               const int size,
                                               const int layer_param_idx,
                                               const int skiplength,
                                               const int i){
  input_gradient[i] = 0.0f;
  const int recurrent_offset = dim - size;
  for(int j = 0; j < size; j++){
    const int w_idx = layer_param_idx + (skiplength * j) + i;

    const float g = gradient[j];
    const float d = differentiate(output[j], logistic);
    const float w = params[w_idx + 1];

    float r;
    if(use_future_gradient)
      r = future_input_gradient[recurrent_offset + j];
    else
      r = 0;

    input_gradient[i] += (g + r) * d * w;
  }
}

static void agnostic_rnn_parameter_gradient_kernel(__mem_ro float *gradient,
                                                   __mem_ro float *output,
                                                   __mem_ro float *future_input_gradient,
                                                   __mem_ro float *previous_output,
                                                   __mem_ro float *input,
                                                   __mem_rw float *param_grad,
                                                   const Nonlinearity logistic,
                                                   const int use_future_gradient,
                                                   const int use_past_output,
                                                   const int dim,
                                                   const int size,
                                                   const int layer_param_idx,
                                                   const int skiplength,
																									 const int abs_grad,
                                                   const int i){
  const int recurrent_offset = dim - size;
  const int w_base = layer_param_idx + (skiplength * i);

  const float g = gradient[i];
  const float d = differentiate(output[i], logistic);

  float l, r;
  if(use_future_gradient)
    r = future_input_gradient[recurrent_offset + i];
  else
    r = 0;

  for(int j = 0; j < dim; j++){
    const int w_idx = w_base + j + 1;

    const float x = input[j];

    if(j < recurrent_offset){
			float update = (g + r) * d * x;
			if(abs_grad && update < 0)
				param_grad[w_idx] -= update;
			else
				param_grad[w_idx] += update;
		}else{
			float update = 0;
      if(use_past_output){
        l = previous_output[j - recurrent_offset];
				update = (g + r) * d * l;
			}
			if(abs_grad && update < 0)
				param_grad[w_idx] -= update;
			else
				param_grad[w_idx] += update;
		}
#ifdef SIEKNET_MAX_GRAD
		if(param_grad[w_idx] >  SIEKNET_MAX_GRAD) param_grad[w_idx] =  SIEKNET_MAX_GRAD;
		if(param_grad[w_idx] < -SIEKNET_MAX_GRAD) param_grad[w_idx] = -SIEKNET_MAX_GRAD;
#endif
  }

	if(abs_grad && (g + r) * d < 0)
		param_grad[w_base] -= (g + r) * d;
	else
		param_grad[w_base] += (g + r) * d;

#ifdef SIEKNET_MAX_GRAD
  if(param_grad[w_base] >  SIEKNET_MAX_GRAD) param_grad[w_base] =  SIEKNET_MAX_GRAD;
  if(param_grad[w_base] < -SIEKNET_MAX_GRAD) param_grad[w_base] = -SIEKNET_MAX_GRAD;
#endif
}
/*<<KERNEL END>>*/
#endif
