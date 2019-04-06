/*<<KERNEL START>>*/


__kernel void lstm_forward_kernel(__mem_ro float *input_nonl, 
                                  __mem_ro float *input_gate, 
                                  __mem_ro float *forget_gate, 
                                  __mem_ro float *output_gate, 
                                  __mem_rw float *cell_state,
                                  __mem_ro float *cell_lstate,
                                  __mem_rw float *layer_output){
  agnostic_lstm_forward_kernel(input_nonl, 
                               input_gate, 
                               forget_gate, 
                               output_gate, 
                               cell_state, 
                               cell_lstate, 
                               layer_output, 
                               get_global_id(0));

}

__kernel void lstm_dstate_kernel(__mem_ro float *gradient,
                                 __mem_ro float *state,
                                 __mem_ro float *output_gate_out,
                                 __mem_ro float *future_dstate,
                                 __mem_ro float *future_forget_gate_out,
                                 __mem_ro float *future_input_gradient,
                                 __mem_rw float *dstate,
                                 const int recurrent_offset,
                                 const int use_future_grads){
  agnostic_lstm_dstate_kernel(gradient, 
                              state, 
                              output_gate_out, 
                              future_dstate, 
                              future_forget_gate_out, 
                              future_input_gradient, 
                              dstate, 
                              recurrent_offset, 
                              use_future_grads, 
                              get_global_id(0));
}

__kernel void lstm_input_nonl_gradient_kernel(__mem_ro float *dstate,
                                              __mem_ro float *input_gate_out,
                                              __mem_ro float *input_nonl_out,
                                              __mem_rw float *input_nonl_gradient,
                                              const Nonlinearity gate_fn){
  agnostic_lstm_input_nonl_gradient_kernel(dstate, 
                                           input_gate_out, 
                                           input_nonl_out,
                                           input_nonl_gradient,
                                           gate_fn,
                                           get_global_id(0));
}

__kernel void lstm_forget_gate_gradient_kernel(__mem_ro float *dstate,
                                               __mem_ro float *last_state,
                                               __mem_ro float *forget_gate_out,
                                               __mem_rw float *forget_gate_gradient,
                                               const Nonlinearity gate_fn,
                                               const int use_past_outputs){
  agnostic_lstm_forget_gate_gradient_kernel(dstate,
                                            last_state,
                                            forget_gate_out,
                                            forget_gate_gradient,
                                            gate_fn,
                                            use_past_outputs,
                                            get_global_id(0));
}

__kernel void lstm_output_gate_gradient_kernel(__mem_ro float *gradient,
                                               __mem_ro float *state,
                                               __mem_ro float *output_gate_out,
                                               __mem_ro float *future_input_gradient,
                                               __mem_rw float *output_gate_gradient,
                                               const Nonlinearity gate_fn,
                                               const int recurrent_offset,
                                               const int use_future_grads){
  agnostic_lstm_output_gate_gradient_kernel(gradient,
                                            state,
                                            output_gate_out,
                                            future_input_gradient,
                                            output_gate_gradient,
                                            gate_fn,
                                            recurrent_offset,
                                            use_future_grads,
                                            get_global_id(0));
}

__kernel void lstm_input_gradient_kernel(__mem_ro float *input_nonl_grad,
                                         __mem_ro float *input_gate_grad,
                                         __mem_ro float *forget_gate_grad,
                                         __mem_ro float *output_gate_grad,
                                         __mem_ro float *params,
                                         __mem_rw float *input_gradient,
                                         const int size,
                                         const int input_dimension,
                                         const int layer_param_offset,
                                         const int skipdist){
  agnostic_lstm_input_gradient_kernel(input_nonl_grad, 
                                      input_gate_grad, 
                                      forget_gate_grad, 
                                      output_gate_grad, 
                                      params, 
                                      input_gradient, 
                                      size, 
                                      input_dimension, 
                                      layer_param_offset, 
                                      skipdist, 
                                      get_global_id(0));
}

__kernel void lstm_parameter_gradient_kernel(__mem_ro float *input_nonl_grad,
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
                                             const int skipdist){
  agnostic_lstm_parameter_gradient_kernel(input_nonl_grad,
                                          input_gate_grad,
                                          forget_gate_grad,
                                          output_gate_grad,
                                          future_input_nonl_grad,
                                          future_input_gate_grad,
                                          future_forget_gate_grad,
                                          future_output_gate_grad,
                                          param_grad,
                                          input,
                                          output,
                                          use_future_grads,
                                          size,
                                          input_dimension,
                                          layer_param_offset,
                                          skipdist,
                                          get_global_id(0));
}

/*<<KERNEL END>>*/
