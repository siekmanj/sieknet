/*<<KERNEL START>>*/
__kernel void rnn_forward_kernel(__constant float *x, // inputs
																 __constant float *r, // recurrent inputs
																 __global float *z, // linear output
																 __constant float *params, //parameters
																 const int dim, 
																 const int size,
																 const int layer_param_idx, 
																 const int skiplength){
	const int i = get_global_id(0);
	z[i] = 0.0f;
	const int w_idx = layer_param_idx + (skiplength * i);
	float sum = 0.0f;
	for(int j = 0; j < dim-size; j++){
		sum += x[j] * params[w_idx + j + 1]; //weights
	}
	for(int j = 0; j < size; j++){
		sum += r[j] * params[w_idx + (dim-size) + j + 1]; //recurrent weights
	}
	z[i] = sum + params[w_idx]; //wx + b
}
/*<<KERNEL END>>*/
