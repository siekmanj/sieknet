/*<<KERNEL START>>*/

__kernel void logistic_kernel(__mem_rw float *x, __mem_rw float *y, Nonlinearity n){
  const int i = get_global_id(0);
	y[i] = activate(x[i], n);
}

__kernel void softmax_sum_kernel(__mem_rw float *z, __mem_rw float *sum, int dim){
  agnostic_softmax_sum_kernel(z, sum, dim);
}

__kernel void softmax_kernel(__mem_ro float *z, __mem_rw float *y, __mem_ro float *sum){
  const int i = get_global_id(0);
  agnostic_softmax_kernel(z, y, sum, i);
}

__kernel void zero_init_kernel(__mem_rw float *x){
  x[get_global_id(0)] = 0.0f;
}

__kernel void cost_kernel(__mem_ro float *o, __mem_ro float *y, __mem_rw float *sum, int dim, Costfn c){
  agnostic_cost_kernel(o, y, sum, dim, c);
}

__kernel void cost_gradient_kernel(__mem_rw float *o, __mem_ro float *y, __mem_rw float *dest, Costfn c){
  const int i = get_global_id(0);
  dest[i] = cost_gradient(o[i], y[i], c);
}

/*<<KERNEL END>>*/
