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

__kernel void mlp_backward_kernel(
																	__global float *grads, 
																	__global float *input, 
																	__global float *output, 
																	__global float *dest, 
																	__global float *params, 
																	__global float *param_grads, 
																	Nonlinearity nonlinearity_type, 
																	int layer_param_idx,
																	int neurons, 
																	int dim
																){
	const int i = get_global_id(0);

	dest[i] = 0;
	for(int j = 0; j < neurons; j++){
		const int w_idx = layer_param_idx + ((dim + 1) * j) + i;
		float w = params[w_idx+1];
		float d = differentiate(output[j], nonlinearity_type);
		float g = grads[j];
		float x = input[i];
		dest[i] += w * d * g;
		param_grads[w_idx+1] = x * d * g;
	}
}

/*<<KERNEL END>>*/
