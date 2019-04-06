#ifndef RNN_H
#define RNN_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// some magic to allow arbitrary numbers of parameters
//#define create_rnn(...) rnn_from_arr((size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))

/*<<KERNEL START>>*/
#define agnostic_rnn_forward_kernel(x, r, z, params, dim, size, layer_param_idx, skiplength, i) \
  {                                                          \
    z[i] = 0.0f;                                             \
    const int w_idx = layer_param_idx + (skiplength * i);    \
    for(int xyz = 0; xyz < dim-size; xyz++)                  \
      z[i] += x[xyz] * params[w_idx + xyz + 1];              \
    for(int xyz = 0; xyz < size; xyz++)                      \
      z[i] += r[xyz] * params[w_idx + (dim-size) + xyz + 1]; \
    z[i] += params[w_idx];                                   \
  }                                                          \
  no_op()

/*<<KERNEL END>>*/
#endif
