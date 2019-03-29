/*<<KERNEL START>>*/
__kernel void logistic_kernel(__global float *x, __global float *y, Nonlinearity n){
	const int i = get_global_id(0);
	y[i] = activate(x[i], n);
}

__kernel void softmax_sum_kernel(__global float *z, __global float *sum, int dim){
	sum[0] = 0.0f;
	float fmax;
	for(int i = 0; i < dim; i++)
		if(z[i] > fmax) fmax = z[i];

	for(int i = 0; i < dim; i++){
		sum[0] += exp(z[i]-fmax);
		z[i] = z[i] - fmax;
	}
}

__kernel void softmax_kernel(__global float *z, __global float *y, __global float *sum){
	const int i = get_global_id(0);
	y[i] = exp(z[i]) / sum[0];
}

__kernel void zero_init_kernel(__global float *x){
	x[get_global_id(0)] = 0.0f;
}

/*<<KERNEL END>>*/
