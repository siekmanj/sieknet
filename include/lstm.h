#ifndef LSTM_H
#define LSTM_H

#include <stdlib.h>
#include <stdio.h>
#include <mlp.h>

#ifdef SIEKNET_USE_GPU
#include <CL/cl.h>
#endif

#define create_lstm(...) lstm_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct lstm_layer{
#ifndef SIEKNET_USE_GPU

  float **input_nonl_z;
  float **input_gate_z;
  float **forget_gate_z;
  float **output_gate_z;

  float **input_nonl_output;
  float **input_gate_output;
  float **forget_gate_output;
  float **output_gate_output;

  float **input_nonl_gradient;
  float **input_gate_gradient;
  float **forget_gate_gradient;
  float **output_gate_gradient;

  float **input_gradient;
  float **cell_gradient;

  float **cell_state;
  float **cell_dstate;

  float **input;
  float **output;

  float *loutput;
  float *lstate;
#else
  cl_mem *input_nonl_z;
  cl_mem *input_gate_z;
  cl_mem *forget_gate_z;
  cl_mem *output_gate_z;

  cl_mem *input_nonl_output;
  cl_mem *input_gate_output;
  cl_mem *forget_gate_output;
  cl_mem *output_gate_output;

  cl_mem *input_nonl_gradient;
  cl_mem *input_gate_gradient;
  cl_mem *forget_gate_gradient;
  cl_mem *output_gate_gradient;

  cl_mem *input_gradient;
  cl_mem *cell_gradient;

  cl_mem *cell_state;
  cl_mem *cell_dstate;

  cl_mem *input;
  cl_mem *output;

  cl_mem loutput;
  cl_mem lstate;
#endif
  int param_offset;

  size_t input_dimension;
  size_t size;
} LSTM_layer;

typedef struct lstm{
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

  LSTM_layer *layers;
  MLP_layer output_layer;
  Costfn cost_fn;
} LSTM;

LSTM lstm_from_arr(size_t *, size_t);
LSTM load_lstm(const char *);
void save_lstm(LSTM *n, const char *);

void lstm_forward(LSTM *, float *);
void lstm_backward(LSTM *);
void lstm_abs_backward(LSTM *);
float lstm_cost(LSTM *, float *);

void lstm_wipe(LSTM *);

void dealloc_lstm(LSTM *);

/*
 * In this file, lstm kernels are implemented as static functions to be used in both the gpu and cpu
 * implementation of sieknet.
 *
 * This was a design decision to emphasize re-use of code, and enforce under-the-hood
 * homogeneity across CPU/GPU implementations. Unfortunately, OpenCL does not allow address
 * space changes (i.e., passing a __global pointer to a function that takes a pointer), which
 * necessitated the use of macros to provide an implementation that could be reused on the 
 * GPU as well as the CPU. When compiled with OpenCL, __mem_ro and __mem_rw are defined as
 * either __constant or const __global, and __global. When compiled on the host machine,
 * mem_rw is defined as an empty macro, and __mem_ro is defined as const. This allows the same
 * code to be reused.
 */

#define __mem_rw
#define __mem_ro const

/* Code above this line will be ignored by the OpenCL compiler */
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


static void agnostic_lstm_forward_kernel(__mem_ro float *input_nonl, 
    __mem_ro float *input_gate, 
    __mem_ro float *forget_gate, 
    __mem_ro float *output_gate, 
    __mem_rw float *cell_state,
    __mem_ro float *cell_lstate,
    __mem_rw float *layer_output,
    int i){
  if(cell_lstate[i] > SIEKNET_MAX_STATE)                                                
    cell_state[i] = input_nonl[i] * input_gate[i] + forget_gate[i] * SIEKNET_MAX_STATE; 
  else if(cell_lstate[i] < -SIEKNET_MAX_STATE)                                          
    cell_state[i] = input_nonl[i] * input_gate[i] - forget_gate[i] * SIEKNET_MAX_STATE; 
  else                                                                                  
    cell_state[i] = input_nonl[i] * input_gate[i] + forget_gate[i] * cell_lstate[i];   
  layer_output[i] = HYPERTAN(cell_state[i]) * output_gate[i];                          
}

static void agnostic_lstm_dstate_kernel(__mem_ro float *gradient,
    __mem_ro float *state,
    __mem_ro float *output_gate_out,
    __mem_ro float *future_dstate,
    __mem_ro float *future_forget_gate_out,
    __mem_ro float *future_input_gradient,
    __mem_rw float *dstate,
    const int recurrent_offset,
    const int use_future_grads, 
    const int i){
  float cell_grad, next_dstate, next_forget;                                                                  
  if(use_future_grads){                                                                                       
    cell_grad = gradient[i] + future_input_gradient[recurrent_offset + i];                                    
    next_dstate = future_dstate[i];                                                                           
    next_forget = future_forget_gate_out[i];                                                                  
  }else{
    cell_grad = gradient[i];
    next_dstate = 0.0f;
    next_forget = 0.0f;
  }
  dstate[i] = cell_grad * output_gate_out[i] * D_HYPERTAN(HYPERTAN(state[i])) + next_dstate * next_forget;
}

static void agnostic_lstm_input_nonl_gradient_kernel(__mem_ro float *dstate,
    __mem_ro float *input_gate_out,
    __mem_ro float *input_nonl_out,
    __mem_rw float *input_nonl_gradient,
    const Nonlinearity gate_fn,
    const int i){
  input_nonl_gradient[i] = dstate[i] * input_gate_out[i] * differentiate(input_nonl_out[i], gate_fn);
}

static void agnostic_lstm_forget_gate_gradient_kernel(__mem_ro float *dstate,
    __mem_ro float *last_state,
    __mem_ro float *forget_gate_out,
    __mem_rw float *forget_gate_gradient,
    const Nonlinearity gate_fn,
    const int use_past_outputs,
    const int i){
  if(use_past_outputs)
    forget_gate_gradient[i] = dstate[i] * last_state[i] * differentiate(forget_gate_out[i], gate_fn);
  else
    forget_gate_gradient[i] = 0.0f;
}

static void agnostic_lstm_output_gate_gradient_kernel(__mem_ro float *gradient,
    __mem_ro float *state,
    __mem_ro float *output_gate_out,
    __mem_ro float *future_input_gradient,
    __mem_rw float *output_gate_gradient,
    const Nonlinearity gate_fn,
    const int recurrent_offset,
    const int use_future_grads,
    const int i){
  float cell_grad;
  if(use_future_grads)
    cell_grad = gradient[i] + future_input_gradient[recurrent_offset + i];
  else
    cell_grad = gradient[i];

  output_gate_gradient[i] = cell_grad * HYPERTAN(state[i]) * differentiate(output_gate_out[i], gate_fn);
}

static void agnostic_lstm_input_gradient_kernel(__mem_ro float *input_nonl_grad,
    __mem_ro float *input_gate_grad,
    __mem_ro float *forget_gate_grad,
    __mem_ro float *output_gate_grad,
    __mem_ro float *params,
    __mem_rw float *input_gradient,
    const int size,
    const int input_dimension,
    const int layer_param_offset,
    const int skipdist,
    const int i){
  input_gradient[i] = 0.0f;
  for(int j = 0; j < size; j++){
    const int params_per_gate = input_dimension+1;
    const int w_idx = layer_param_offset + (skipdist * j) + i;

    input_gradient[i] += input_nonl_grad[j]  * params[w_idx + 0 * params_per_gate + 1];
    input_gradient[i] += input_gate_grad[j]  * params[w_idx + 1 * params_per_gate + 1];
    input_gradient[i] += forget_gate_grad[j] * params[w_idx + 2 * params_per_gate + 1];
    input_gradient[i] += output_gate_grad[j] * params[w_idx + 3 * params_per_gate + 1];
  }
}
static void agnostic_lstm_parameter_gradient_kernel(__mem_ro float *input_nonl_grad,
    __mem_ro float *input_gate_grad,
    __mem_ro float *forget_gate_grad,
    __mem_ro float *output_gate_grad,
    __mem_ro float *future_input_nonl_grad,
    __mem_ro float *future_input_gate_grad,
    __mem_ro float *future_forget_gate_grad,
    __mem_ro float *future_output_gate_grad,
    __mem_rw float *param_grad,
    __mem_ro float *input,
    __mem_ro float *output,
    const int use_future_grads,
    const int size,
    const int input_dimension,
    const int layer_param_offset,
    const int skipdist,
    const int abs_grad,
    const int i){

  const int recurrent_offset = input_dimension - size;
  const int params_per_gate = input_dimension+1; 
  const int w_idx = layer_param_offset + (skipdist * i); //cell param offset

  for(int j = 0; j < input_dimension; j++){
    const int aw_idx = w_idx + 0 * params_per_gate + 1 + j;
    const int iw_idx = w_idx + 1 * params_per_gate + 1 + j;
    const int fw_idx = w_idx + 2 * params_per_gate + 1 + j;
    const int ow_idx = w_idx + 3 * params_per_gate + 1 + j;

    float input_nonl_update  = 0;
    float input_gate_update  = 0;
    float forget_gate_update = 0;
    float output_gate_update = 0;

    if(j < recurrent_offset){
      input_nonl_update  = input_nonl_grad[i]  * input[j];
      input_gate_update  = input_gate_grad[i]  * input[j];
      forget_gate_update = forget_gate_grad[i] * input[j];
      output_gate_update = output_gate_grad[i] * input[j];

    }else if(use_future_grads){
      input_nonl_update  = future_input_nonl_grad[i]  * output[j - recurrent_offset];
      input_gate_update  = future_input_gate_grad[i]  * output[j - recurrent_offset];
      forget_gate_update = future_forget_gate_grad[i] * output[j - recurrent_offset];
      output_gate_update = future_output_gate_grad[i] * output[j - recurrent_offset];
    }
    if(abs_grad){

      if(input_nonl_update < 0)
        param_grad[aw_idx] -= input_nonl_update;
      else
        param_grad[aw_idx] += input_nonl_update;

      if(input_gate_update < 0)
        param_grad[iw_idx] -= input_gate_update;
      else
        param_grad[iw_idx] += input_gate_update;

      if(forget_gate_update < 0)
        param_grad[fw_idx] -= forget_gate_update;
      else
        param_grad[fw_idx] += forget_gate_update;
      
      if(output_gate_update < 0)
        param_grad[ow_idx] -= output_gate_update;
      else
        param_grad[ow_idx] += output_gate_update;

    }else{

        param_grad[aw_idx] += input_nonl_update;
        param_grad[iw_idx] += input_gate_update;
        param_grad[fw_idx] += forget_gate_update;
        param_grad[ow_idx] += output_gate_update;

    }
#ifdef SIEKNET_MAX_GRAD
    if(param_grad[aw_idx] >  SIEKNET_MAX_GRAD) param_grad[aw_idx] =  SIEKNET_MAX_GRAD;
    if(param_grad[aw_idx] < -SIEKNET_MAX_GRAD) param_grad[aw_idx] = -SIEKNET_MAX_GRAD;
    if(param_grad[iw_idx] >  SIEKNET_MAX_GRAD) param_grad[iw_idx] =  SIEKNET_MAX_GRAD;
    if(param_grad[iw_idx] < -SIEKNET_MAX_GRAD) param_grad[iw_idx] = -SIEKNET_MAX_GRAD;
    if(param_grad[fw_idx] >  SIEKNET_MAX_GRAD) param_grad[fw_idx] =  SIEKNET_MAX_GRAD;
    if(param_grad[fw_idx] < -SIEKNET_MAX_GRAD) param_grad[fw_idx] = -SIEKNET_MAX_GRAD;
    if(param_grad[ow_idx] >  SIEKNET_MAX_GRAD) param_grad[ow_idx] =  SIEKNET_MAX_GRAD;
    if(param_grad[ow_idx] < -SIEKNET_MAX_GRAD) param_grad[ow_idx] = -SIEKNET_MAX_GRAD;
#endif
  }
  const int ab_idx = w_idx + 0 * params_per_gate;
  const int ib_idx = w_idx + 1 * params_per_gate;
  const int fb_idx = w_idx + 2 * params_per_gate;
  const int ob_idx = w_idx + 3 * params_per_gate;

  if(abs_grad){

    if(input_nonl_grad[i] < 0)
      param_grad[ab_idx] -= input_nonl_grad[i];
    else
      param_grad[ab_idx] += input_nonl_grad[i];

    if(input_gate_grad[i] < 0)
      param_grad[ib_idx] -= input_gate_grad[i];
    else
      param_grad[ib_idx] += input_gate_grad[i];

    if(forget_gate_grad[i] < 0)
      param_grad[fb_idx] -= forget_gate_grad[i];
    else
      param_grad[fb_idx] += forget_gate_grad[i];

    if(output_gate_grad[i] < 0)
      param_grad[ob_idx] -= output_gate_grad[i];
    else
      param_grad[ob_idx] += output_gate_grad[i];

  }else{
    param_grad[ab_idx] += input_nonl_grad[i];
    param_grad[ib_idx] += input_gate_grad[i];
    param_grad[fb_idx] += forget_gate_grad[i];
    param_grad[ob_idx] += output_gate_grad[i];
  }
#ifdef SIEKNET_MAX_GRAD
  if(param_grad[ab_idx] >  SIEKNET_MAX_GRAD) param_grad[ab_idx] =  SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] < -SIEKNET_MAX_GRAD) param_grad[ab_idx] = -SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] >  SIEKNET_MAX_GRAD) param_grad[ab_idx] =  SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] < -SIEKNET_MAX_GRAD) param_grad[ab_idx] = -SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] >  SIEKNET_MAX_GRAD) param_grad[ab_idx] =  SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] < -SIEKNET_MAX_GRAD) param_grad[ab_idx] = -SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] >  SIEKNET_MAX_GRAD) param_grad[ab_idx] =  SIEKNET_MAX_GRAD;
  if(param_grad[ab_idx] < -SIEKNET_MAX_GRAD) param_grad[ab_idx] = -SIEKNET_MAX_GRAD;
#endif
}
/*<<KERNEL END>>*/
#endif
