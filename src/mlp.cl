/*<<KERNEL START>>*/

__kernel void mlp_forward_kernel(__global float *x, 
                                 __global float *z, 
                                 __global float *params, 
                                 const int dim, 
                                 const int layer_param_idx, 
                                 const int skiplength){
  const int i = get_global_id(0);
  /*
  z[i] = 0.0f;
  const int w_idx = layer_param_idx + (skiplength * i);
  float sum = 0.0f;
  for(int j = 0; j < dim; j++){
    sum += x[j] * params[w_idx + j + 1]; //weights
  }
  z[i] = sum + params[w_idx]; //wx + b
  */
  agnostic_mlp_forward_kernel(x, z, params, dim, layer_param_idx, skiplength, i);
}

__kernel void mlp_input_gradient_kernel(__global float *grads,
                                        __global float *output,
                                        __global float *params,
                                        __global float *dest,
                                        const Nonlinearity nonlinearity_type,
                                        const int layer_param_idx,
                                        const int size,
                                        const int dim){
  const int i = get_global_id(0); //ith input gradient
  /*
  dest[i] = 0.0f;
  for(int j = 0; j < size; j++){
    const int w_idx = layer_param_idx + ((dim + 1) * j) + i;
    float w = params[w_idx+1];
    float d = differentiate(output[j], nonlinearity_type);
    float g = grads[j];
    dest[i] += w * d * g;
  }
  */
  agnostic_mlp_input_gradient_kernel(grads, output, params, dest, nonlinearity_type, layer_param_idx, size, dim, i);

}

__kernel void mlp_parameter_gradient_kernel(__global float *grads,
                                            __global float *output,
                                            __global float *input,
                                            __global float *param_grad,
                                            const Nonlinearity nonlinearity_type,
                                            const int layer_param_idx,
                                            const int size,
                                            const int dim){
  const int i = get_global_id(0); //ith neuron of current layer
  /*
  float d = differentiate(output[i], nonlinearity_type);
  float g = grads[i];

  const int w_idx = layer_param_idx + ((dim + 1) * i);
  param_grad[w_idx] += d * g; //bias grad

  for(int j = 0; j < dim; j++){
    float x = input[j];
    param_grad[w_idx+j+1] += x * d * g; //weight grads
  }
  */
  agnostic_mlp_parameter_gradient_kernel(grads, output, input, param_grad, nonlinearity_type, layer_param_idx, size, dim, i);
}

/*<<KERNEL END>>*/
