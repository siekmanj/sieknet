/*<<KERNEL START>>*/
__kernel void logistic_kernel(__global float *x, __global float *y, Nonlinearity n){
	const int i = get_global_id(0);
	y[i] = activate(x[i], n);
}
/*<<KERNEL END>>*/
