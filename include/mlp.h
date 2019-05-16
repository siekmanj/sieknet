#ifndef MLP_H
#define MLP_H


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <conf.h>
#include <logistic.h>

#ifdef SIEKNET_USE_GPU
#include <CL/cl.h>
#endif

#define create_mlp(...) mlp_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

typedef struct mlp_layer{
#ifndef SIEKNET_USE_GPU
	float *input_gradient;
	float *z;
	float *output;
	float *input;
#else
  cl_mem input_gradient;
  cl_mem z;
  cl_mem output;
  cl_mem input;
#endif
	int param_offset;
	size_t size;
	size_t input_dimension;
	Nonlinearity logistic;
} MLP_layer;


typedef struct mlp{
	MLP_layer *layers;
	size_t depth;
	size_t num_params;
	size_t input_dimension;
	size_t output_dimension;
	size_t guess;

	float learning_rate;
	float *output;
  float performance;

#ifndef SIEKNET_USE_GPU
	float *params;
	float *param_grad;

	float *cost_gradient;
#else
	cl_mem params;
	cl_mem param_grad;
	cl_mem network_input;
	cl_mem output_label;

	cl_mem cost_gradient;
#endif
	Costfn cost_fn;
} MLP;

MLP mlp_from_arr(size_t[], size_t);
MLP load_mlp(const char *);
MLP *copy_mlp(MLP *);

#ifndef SIEKNET_USE_GPU
MLP_layer cpu_create_MLP_layer(const size_t, const size_t, float *, const int, const Nonlinearity);
void cpu_mlp_layer_forward(MLP_layer *, float *, float *);
void cpu_mlp_layer_backward(MLP_layer *, float *, float *, float *, int);
float cpu_cost(const float *, const float *, float *, size_t, Costfn);
void cpu_zero_2d_arr(float **, size_t, size_t);
#else
MLP_layer gpu_create_MLP_layer(size_t, size_t, cl_mem, int, Nonlinearity);
void gpu_mlp_layer_forward(MLP_layer *, cl_mem, cl_mem);
void gpu_mlp_layer_backward(MLP_layer *, cl_mem, cl_mem, cl_mem, int);
float gpu_cost(cl_mem, cl_mem, cl_mem, size_t, Costfn);
void gpu_zero_2d_arr(cl_mem *, size_t, size_t);
#endif

void mlp_forward(MLP *, float *);
void mlp_backward(MLP *);
void mlp_abs_backward(MLP *);
float mlp_cost(MLP *, float *);

void save_mlp(MLP *n, const char* filename);
void dealloc_mlp(MLP *);

void xavier_init(float *, size_t, size_t);
void zero_2d_arr(float **, size_t, size_t);

int argmax(float *, size_t);
int sample_softmax(float *, size_t);

float **alloc_2d_array(size_t, size_t);

#ifdef SIEKNET_USE_GPU
void mlp_kernel_setup();
#endif

/*
 * In this file, mlp kernels are implemented as macros to be used in both the gpu and cpu
 * implementation of Sieknet.
 * This was a design decision to emphasize re-use of code, and enforce under-the-hood
 * homogeneity across implementations. Unfortunately, OpenCL does not allow address space
 * changes (i.e., passing a __global pointer to a function that takes a pointer), which
 * necessitated the use of macros to provide an implementation that could be reused on the 
 * GPU as well as the CPU.
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

static void agnostic_mlp_forward_kernel(__mem_ro float *x,
                                        __mem_rw float *z,
                                        __mem_ro float *params,
                                        const int dim,
                                        const int layer_param_idx,
                                        const int skiplength,
                                        const int i){
	z[i] = 0.0f;                                          
	const int w_idx = layer_param_idx + (skiplength * i); 
	float sum = 0.0f;                                     
	for(int j = 0; j < dim; j++)                          
		sum += x[j] * params[w_idx + j + 1];                
	z[i] = sum + params[w_idx];                           
}

static void agnostic_mlp_input_gradient_kernel(__mem_ro float *grads,
                                               __mem_ro float *output,
                                               __mem_ro float *params,
                                               __mem_rw float *dest,
                                               const Nonlinearity nonlinearity_type,
                                               const int layer_param_idx,
                                               const int size,
                                               const int dim, 
                                               const int i){
	dest[i] = 0.0f;                                            
	for(int j = 0; j < size; j++){                             
		const int w_idx = layer_param_idx + ((dim + 1) * j) + i;
		float w = params[w_idx+1];                             
		float d = differentiate(output[j], nonlinearity_type);   
		float g = grads[j];                                      
		dest[i] += w * d * g;                                    
	}                                                          
}

static void agnostic_mlp_parameter_gradient_kernel(__mem_ro float *grads,
                                                   __mem_ro float *output,
                                                   __mem_ro float *input,
                                                   __mem_rw float *param_grad,
                                                   const Nonlinearity nonlinearity_type,
                                                   const int layer_param_idx,
                                                   const int size,
                                                   const int dim,
																									 const int abs_grad,
                                                   const int i){
	const float d = differentiate(output[i], nonlinearity_type);
	const float g = grads[i];
	const int w_idx = layer_param_idx + ((dim + 1) * i);

	for(int j = 0; j < dim; j++){
		float x = input[j];
		float w_update = x * d * g;

		if(abs_grad && w_update < 0)
			param_grad[w_idx + j + 1] -= w_update;
		else
			param_grad[w_idx + j + 1] += w_update;
	}                                                      

	float b_update = 1 * d * g;
	if(abs_grad && b_update < 0)
		param_grad[w_idx] -= b_update;
	else
		param_grad[w_idx] += b_update;
}

/*<<KERNEL END>>*/
#endif
