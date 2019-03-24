/*<<KERNEL START>>*/
__kernel void lstm_forward_kernel(__global float *input_nonl, 
																	__global float *input_gate, 
																	__global float *forget_gate, 
																	__global float *output_gate, 
																	__global float *cell_state,
																	__global float *cell_lstate,
																	__global float *layer_output){
	const int i = get_global_id(0);
	cell_state[i] = input_nonl[i] * input_gate[i] + forget_gate[i] * cell_lstate[i];
	layer_output[i] = HYPERTAN(cell_state[i]) * output_gate[i];
}


/*<<KERNEL END>>*/
