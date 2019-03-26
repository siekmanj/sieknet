/*<<KERNEL START>>*/
__kernel void lstm_forward_kernel(__global float *input_nonl, 
																	__global float *input_gate, 
																	__global float *forget_gate, 
																	__global float *output_gate, 
																	__global float *cell_state,
																	__global float *cell_lstate,
																	__global float *layer_output){
	const int i = get_global_id(0);
	agnostic_lstm_forward_kernel(input_nonl, input_gate, forget_gate, output_gate, cell_state, cell_lstate, layer_output, i);

}
__kernel void lstm_internal_gradient_kernel(__global float *input_nonl_out,
																					  __global float *input_gate_out,
																					  __global float *forget_gate_out,
																					  __global float *future_forget_gate_out,
																					  __global float *output_gate_out,
 
 																					  __global float *input_nonl_grad,
 					 	 															  __global float *input_gate_grad,
																					  __global float *forget_gate_grad,
																					  __global float *output_gate_grad,
																					 
																					  __global float *gradient,
																					  __global float *state,
																					  __global float *dstate,
																					  __global float *future_dstate,
																					  __global float *last_state,

																					  __global float *input_gradient,
																						__global float *future_input_gradient,

																					  Nonlinearity gate_fn,
																					  Nonlinearity nonl_fn,

																						int recurrent_offset,
																					  int use_future_grads,
																					  int use_past_outputs){
	const int i = get_global_id(0);

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

	input_nonl_grad[i] = dstate[i] * input_gate_out[i] * differentiate(input_nonl_out[i], nonl_fn);
	input_gate_grad[i] = dstate[i] * input_nonl_out[i] * differentiate(input_gate_out[i], gate_fn);
	if(use_past_outputs) 
		forget_gate_grad[i] = dstate[i] * last_state[i] * differentiate(forget_gate_out[i], gate_fn);
	else
		forget_gate_grad[i] = 0.0f;
	output_gate_grad[i] = cell_grad * HYPERTAN(state[i]) * differentiate(output_gate_out[i], gate_fn);
}

__kernel void lstm_input_gradient_kernel(__global float *input_nonl_grad,
																				 __global float *input_gate_grad,
																				 __global float *forget_gate_grad,
																				 __global float *output_gate_grad,
																				 __global float *params,
																				 __global float *input_gradient,
																				 int size,
																				 int input_dimension,
																				 int layer_param_offset,
																				 int skipdist){
	const int i = get_global_id(0);
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

__kernel void lstm_parameter_gradient_kernel(__global float *input_nonl_grad,
																						 __global float *input_gate_grad,
																						 __global float *forget_gate_grad,
																						 __global float *output_gate_grad,
																						 __global float *future_input_nonl_grad,
																						 __global float *future_input_gate_grad,
																						 __global float *future_forget_gate_grad,
																						 __global float *future_output_gate_grad,
																						 __global float *param_grad,
																						 __global float *input,
																						 __global float *output,
																						 int use_future_grads,
																						 int size,
																						 int input_dimension,
																						 int layer_param_offset,
																						 int skipdist){
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
#ifdef MAX_GRAD
		if(param_grad[aw_idx] >  MAX_GRAD) param_grad[aw_idx] =  MAX_GRAD;
		if(param_grad[aw_idx] < -MAX_GRAD) param_grad[aw_idx] = -MAX_GRAD;
		if(param_grad[iw_idx] >  MAX_GRAD) param_grad[iw_idx] =  MAX_GRAD;
		if(param_grad[iw_idx] < -MAX_GRAD) param_grad[iw_idx] = -MAX_GRAD;
		if(param_grad[fw_idx] >  MAX_GRAD) param_grad[fw_idx] =  MAX_GRAD;
		if(param_grad[fw_idx] < -MAX_GRAD) param_grad[fw_idx] = -MAX_GRAD;
		if(param_grad[ow_idx] >  MAX_GRAD) param_grad[ow_idx] =  MAX_GRAD;
		if(param_grad[ow_idx] < -MAX_GRAD) param_grad[ow_idx] = -MAX_GRAD;
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
#ifdef MAX_GRAD
	if(param_grad[ab_idx] >  MAX_GRAD) param_grad[ab_idx] =  MAX_GRAD;
	if(param_grad[ab_idx] < -MAX_GRAD) param_grad[ab_idx] = -MAX_GRAD;
	if(param_grad[ab_idx] >  MAX_GRAD) param_grad[ab_idx] =  MAX_GRAD;
	if(param_grad[ab_idx] < -MAX_GRAD) param_grad[ab_idx] = -MAX_GRAD;
	if(param_grad[ab_idx] >  MAX_GRAD) param_grad[ab_idx] =  MAX_GRAD;
	if(param_grad[ab_idx] < -MAX_GRAD) param_grad[ab_idx] = -MAX_GRAD;
	if(param_grad[ab_idx] >  MAX_GRAD) param_grad[ab_idx] =  MAX_GRAD;
	if(param_grad[ab_idx] < -MAX_GRAD) param_grad[ab_idx] = -MAX_GRAD;
#endif
}

/*<<KERNEL END>>*/
