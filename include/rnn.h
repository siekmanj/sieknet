#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define __mem_rw
#define __mem_ro const

/*<<KERNEL START>>*/

#if !defined(__mem_rw)
#define __mem_rw __mem_rw
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

/*<<KERNEL END>>*/
#endif
