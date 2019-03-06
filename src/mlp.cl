/*<<KERNEL START>>*/

__kernel void mlp_forward_kernel(__global float *x, __global float *z, __global float *params, int dim, int layer_param_idx){
	const int i = get_global_id(0);
	z[i] = 0;
	const int w_idx = layer_param_idx + ((dim + 1) * i);
	float sum = 0;
	for(int j = 0; j < dim; j++){
		sum += x[j] * params[w_idx + j + 1]; //weights
	}
	z[i] = sum + params[w_idx]; //wx + b
}

__kernel void mlp_input_gradient_kernel(
																				__global float *grads,
																				__global float *output,
																				__global float *params,
																				__global float *dest,
																				int layer_param_idx,
																				int neurons,
																				int dim
																			 ){
	const int i = get_global_id(0); //ith input gradient
	dest[i] = 0;
	for(int j = 0; j < neurons; j++){
		const int w_idx = layer_param_idx + ((dim + 1) * j) + i;
		float w = params[w_idx+1];
		float d = differentiate(output[j], nonlinearity_type);
		float g = grads[j];
		dest[i] += w * d * g;
	}

}

__kernel void mlp_parameter_gradient_kernel(
																						__global float *grads,
																						__global float *output,
																						__global float *input,
																						__global float *param_grad,
																						Nonlinearity nonlinearity_type,
																						int layer_param_idx,
																						int neurons,
																						int dim
																					 ){
	const int i = get_global_id(0); //ith neuron of current layer

	float d = differentiate(output[i], nonlinearity_type);
	float g = grads[i];

	const int w_idx = layer_param_idx + ((dim + 1) * i);
	param_grad[w_idx] = d * g; //set bias grad

	for(int j = 1; j < dim + 1; j++){
		float x = input[j];
		param_grads[w_idx+1] = x * d * g; //set weight grads
	}
}


/*<<KERNEL END>>*/
