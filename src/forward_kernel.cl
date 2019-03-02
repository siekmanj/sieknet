__kernel void linear_kernel(__global float *x, __global float *z, __global float *params, int dim, int layer_param_idx){
	const int i = get_global_id(0);
	z[i] = 0;
	const int w_idx = layer_param_idx + ((dim + 1) * i);
	float sum = 0;
	for(int j = 0; j < dim; j++){
		sum += x[j] * params[w_idx + j + 1]; //weights
	}
	z[i] = sum + params[w_idx]; //wx + b
}

__kernel void sigmoid_kernel(__global float *x, __global float *y){
	const int i = get_global_id(0);
	y[i] = (1/(1+exp(-x[i])));
}

/*
__kernel void relu_kernel(__global float *x, __global float *y){
	const int i = get_global_id(0);
	y[i] = 
}
*/
