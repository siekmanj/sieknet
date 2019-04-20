/*<<KERNEL START>>*/

__kernel void make_onehot_kernel(__global float *dest, const int h){
  const int i = get_global_id(0);
  if(i == h)
    dest[i] = 1.0f;
  else
    dest[i] = 0.0f;
}

__kernel void rnn_forward_kernel(__mem_ro float *x, // inputs
                                 __mem_ro float *r, // recurrent inputs
                                 __mem_rw float *z, // linear output
                                 __mem_ro float *params, //parameters
                                 const int dim, 
                                 const int size,
                                 const int layer_param_idx, 
                                 const int skiplength){
  agnostic_rnn_forward_kernel(x, 
                              r, 
                              z, 
                              params, 
                              dim, 
                              size, 
                              layer_param_idx, 
                              skiplength, 
                              get_global_id(0));
}

__kernel void rnn_input_gradient_kernel(__mem_ro float *gradient,
                                        __mem_ro float *output,
                                        __mem_ro float *params,
                                        __mem_ro float *future_input_gradient,
                                        __mem_rw float *input_gradient,
                                        const Nonlinearity logistic,
                                        const int use_future_gradient,
                                        const int dim,
                                        const int size,
                                        const int layer_param_idx,
                                        const int skiplength){
  agnostic_rnn_input_gradient_kernel(gradient,
                                     output,
                                     params,
                                     future_input_gradient,
                                     input_gradient,
                                     logistic,
                                     use_future_gradient,
                                     dim,
                                     size,
                                     layer_param_idx,
                                     skiplength,
                                     get_global_id(0));
}
__kernel void rnn_parameter_gradient_kernel(__mem_ro float *gradient,
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
                                            const int skiplength){
  agnostic_rnn_parameter_gradient_kernel(gradient,
                                         output,
                                         future_input_gradient,
                                         previous_output,
                                         input,
                                         param_grad,
                                         logistic,
                                         use_future_gradient,
                                         use_past_output,
                                         dim,
                                         size,
                                         layer_param_idx,
                                         skiplength,
                                         get_global_id(0));
}
/*<<KERNEL END>>*/
