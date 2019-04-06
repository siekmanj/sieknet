/*<<KERNEL START>>*/

__kernel void make_onehot_kernel(__global float *dest, const int h){
  const int i = get_global_id(0);
  if(i == h)
    dest[i] = 1.0f;
  else
    dest[i] = 0.0f;
}

__kernel void rnn_forward_kernel(__mem_ro float *x, // inputs
                                 __mem_ro float *r, // recurrent inputs
                                 __mem_rw float *z, // linear output
                                 __mem_ro float *params, //parameters
                                 const int dim, 
                                 const int size,
                                 const int layer_param_idx, 
                                 const int skiplength){
  const int i = get_global_id(0);
  agnostic_rnn_forward_kernel(x, r, z, params, dim, size, layer_param_idx, skiplength, i);
}
/*<<KERNEL END>>*/
