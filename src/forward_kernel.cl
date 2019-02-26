//NO SOFTMAX SUPPORT

__kernel void linear_kernel(__global float *x, __global *z, __global float *params, int dim, int layer_param_idx){
	const int i = get_global_id(0);
	z[i] = 0;
	const int w_idx = layer_param_idx + ((dim + 1) * i);
	for(int j = 0; j < dim; j++){
		z[i] += x[j] * params[param_idx + j + 1]; //weights
	}
	z[i] += params[param_idx]; //bias
}

__kernel void sigmoid_kernel(__global float *x, __global *y){
	const int i = get_global_id(0);
	y[i] = SIGMOID(x[i]);
}
