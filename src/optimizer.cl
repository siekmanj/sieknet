__kernel void sgd_step(__global float *params, __global float *param_grads, float learning_rate){
	const int i = get_global_id();
	params[i] += param_grads[i] * learning_rate;
	param_grads[i] = 0;
}

__kernel void momentum_step(__global float *params, __global float *param_grads, __global float *param_momentum, float alpha, float beta){
	const int i = get_global_id();
	param_momentum[i] = 
	params[i] += param_grads[i] * learning_rate;
	param_grads[i] = 0;
}
