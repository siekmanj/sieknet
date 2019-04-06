/*<<KERNEL START>>*/

#ifdef SIEKNET_AMDGPU_READONLY_SPEEDUP
#define __gpu_ro __constant
#else
#define __gpu_ro const __global
#endif

__kernel void logistic_kernel(__global float *x, __global float *y, Nonlinearity n){
  const int i = get_global_id(0);
	y[i] = activate(x[i], n);
}

__kernel void softmax_sum_kernel(__global float *z, __global float *sum, int dim){
  agnostic_softmax_sum_kernel(z, sum, dim);
}

__kernel void softmax_kernel(__global float *z, __global float *y, __global float *sum){
  const int i = get_global_id(0);
  agnostic_softmax_kernel(z, y, sum, i);
}

__kernel void zero_init_kernel(__global float *x){
  x[get_global_id(0)] = 0.0f;
}

__kernel void cost_kernel(__global float *o, __gpu_ro float *y, __global float *sum, int dim, Costfn c){
  agnostic_cost_kernel(o, y, sum, dim, c);
}

__kernel void cost_gradient_kernel(__global float *o, __gpu_ro float *y, __global float *dest, Costfn c){
  const int i = get_global_id(0);
  agnostic_cost_gradient_kernel(o, y, dest, c, i);
}

/*<<KERNEL END>>*/
