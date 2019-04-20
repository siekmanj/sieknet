/*<<KERNEL START>>*/
__kernel void sgd_step_kernel(__global float *params, __global float *param_grads, float learning_rate){
  const int i = get_global_id(0);
  params[i] -= param_grads[i] * learning_rate;
  param_grads[i] = 0.0f;
}

__kernel void momentum_step_kernel(__global float *params, __global float *param_grads, __global float *param_momentum, float alpha, float beta){
  const int i = get_global_id(0);
  param_momentum[i] = beta * param_momentum[i] + alpha * param_grads[i];
  params[i] -= param_momentum[i];
  param_grads[i] = 0.0f;
}
/*<<KERNEL END>>*/
