/*<<KERNEL START>>*/

#ifdef SIEKNET_AMDGPU_READONLY_SPEEDUP
#define __gpu_ro __constant
#else
#define __gpu_ro const __global
#endif

__kernel void lstm_forward_kernel(__gpu_ro float *input_nonl, 
                                  __gpu_ro float *input_gate, 
                                  __gpu_ro float *forget_gate, 
                                  __gpu_ro float *output_gate, 
                                  __global float *cell_state,
                                  __gpu_ro float *cell_lstate,
                                  __global float *layer_output){
  const int i = get_global_id(0);
  agnostic_lstm_forward_kernel(input_nonl, input_gate, forget_gate, output_gate, cell_state, cell_lstate, layer_output, i);

}

__kernel void lstm_dstate_kernel(__global float *gradient,
                                 __gpu_ro float *state,
                                 __gpu_ro float *output_gate_out,
                                 __gpu_ro float *future_dstate,
                                 __gpu_ro float *future_forget_gate_out,
                                 __gpu_ro float *future_input_gradient,
                                 __global float *dstate,
                                 const int recurrent_offset,
                                 const int use_future_grads){
  const int i = get_global_id(0);
  //agnostic_lstm_dstate_kernel(gradient, state, output_gate_out, future_dstate, future_forget_gate_out, future_input_gradient, dstate, recurrent_offset, use_future_grads, i);
  
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

__kernel void lstm_input_nonl_gradient_kernel(__gpu_ro float *dstate,
                                              __gpu_ro float *input_gate_out,
                                              __gpu_ro float *input_nonl_out,
                                              __global float *input_nonl_gradient,
                                              const Nonlinearity gate_fn){
  const int i = get_global_id(0);
  input_nonl_gradient[i] = dstate[i] * input_gate_out[i] * differentiate(input_nonl_out[i], gate_fn);
}

__kernel void lstm_forget_gate_gradient_kernel(__gpu_ro float *dstate,
                                               __gpu_ro float *last_state,
                                               __gpu_ro float *forget_gate_out,
                                               __global float *forget_gate_gradient,
                                               const Nonlinearity gate_fn,
                                               const int use_past_outputs){
  const int i = get_global_id(0);
  if(use_past_outputs)
    forget_gate_gradient[i] = dstate[i] * last_state[i] * differentiate(forget_gate_out[i], gate_fn);
  else
    forget_gate_gradient[i] = 0.0f;
}

__kernel void lstm_output_gate_gradient_kernel(__gpu_ro float *gradient,
                                               __gpu_ro float *state,
                                               __gpu_ro float *output_gate_out,
                                               __gpu_ro float *future_input_gradient,
                                               __global float *output_gate_gradient,
                                               const Nonlinearity gate_fn,
                                               const int recurrent_offset,
                                               const int use_future_grads){
  const int i = get_global_id(0);
  float cell_grad;
  if(use_future_grads)
    cell_grad = gradient[i] + future_input_gradient[recurrent_offset + i];
  else
    cell_grad = gradient[i];

  output_gate_gradient[i] = cell_grad * HYPERTAN(state[i]) * differentiate(output_gate_out[i], gate_fn);
}

__kernel void lstm_input_gradient_kernel(__gpu_ro float *input_nonl_grad,
                                         __gpu_ro float *input_gate_grad,
                                         __gpu_ro float *forget_gate_grad,
                                         __gpu_ro float *output_gate_grad,
                                         __gpu_ro float *params,
                                         __global float *input_gradient,
                                         const int size,
                                         const int input_dimension,
                                         const int layer_param_offset,
                                         const int skipdist){
  const int i = get_global_id(0);
  agnostic_lstm_input_gradient_kernel(input_nonl_grad, input_gate_grad, forget_gate_grad, output_gate_grad, params, input_gradient, size, input_dimension, layer_param_offset, skipdist, i);
  /*
  input_gradient[i] = 0.0f;
  for(int j = 0; j < size; j++){
    const int params_per_gate = input_dimension+1;
    const int w_idx = layer_param_offset + (skipdist * j) + i;

    input_gradient[i] += input_nonl_grad[j]  * params[w_idx + 0 * params_per_gate + 1];
    input_gradient[i] += input_gate_grad[j]  * params[w_idx + 1 * params_per_gate + 1];
    input_gradient[i] += forget_gate_grad[j] * params[w_idx + 2 * params_per_gate + 1];
    input_gradient[i] += output_gate_grad[j] * params[w_idx + 3 * params_per_gate + 1];
  }
  */
}

__kernel void lstm_parameter_gradient_kernel(__gpu_ro float *input_nonl_grad,
                                             __gpu_ro float *input_gate_grad,
                                             __gpu_ro float *forget_gate_grad,
                                             __gpu_ro float *output_gate_grad,
                                             __gpu_ro float *future_input_nonl_grad,
                                             __gpu_ro float *future_input_gate_grad,
                                             __gpu_ro float *future_forget_gate_grad,
                                             __gpu_ro float *future_output_gate_grad,
                                             __global float *param_grad,
                                             __gpu_ro float *input,
                                             __gpu_ro float *output,
                                             const int use_future_grads,
                                             const int size,
                                             const int input_dimension,
                                             const int layer_param_offset,
                                             const int skipdist){
  const int i = get_global_id(0);
  const int recurrent_offset = input_dimension - size;
  const int params_per_gate = input_dimension+1; 
  const int w_idx = layer_param_offset + (skipdist * i); //cell param offset
  for(int j = 0; j < input_dimension; j++){
    const int aw_idx = w_idx + 0 * params_per_gate + 1 + j;
    const int iw_idx = w_idx + 1 * params_per_gate + 1 + j;
    const int fw_idx = w_idx + 2 * params_per_gate + 1 + j;
    const int ow_idx = w_idx + 3 * params_per_gate + 1 + j;
    
    if(j < recurrent_offset){
      param_grad[aw_idx] += input_nonl_grad[i]  * input[j];
      param_grad[iw_idx] += input_gate_grad[i]  * input[j];
      param_grad[fw_idx] += forget_gate_grad[i] * input[j];
      param_grad[ow_idx] += output_gate_grad[i] * input[j];

    }else if(use_future_grads){
      param_grad[aw_idx] += future_input_nonl_grad[i]  * output[j - recurrent_offset];
      param_grad[iw_idx] += future_input_gate_grad[i]  * output[j - recurrent_offset];
      param_grad[fw_idx] += future_forget_gate_grad[i] * output[j - recurrent_offset];
      param_grad[ow_idx] += future_output_gate_grad[i] * output[j - recurrent_offset];
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

  param_grad[ab_idx] += input_nonl_grad[i];
  param_grad[ib_idx] += input_gate_grad[i];
  param_grad[fb_idx] += forget_gate_grad[i];
  param_grad[ob_idx] += output_gate_grad[i];
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
