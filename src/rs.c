#include <conf.h>
#include <rs.h>
#include <math.h>

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return mean + norm * std;
}

RS create_rs(float *seed, size_t num_params, size_t n){
	RS r;
	r.std = 0.5;
	r.step_size = 0.01;
	r.directions = n;
	r.num_params = num_params;
	r.algo = v1;

	if(seed)
		r.params = seed;
	else
		r.params = calloc(num_params, sizeof(float));

	r.r_pos = ALLOC(float, n);
	r.r_neg = ALLOC(float, n);

	r.perturbances = ALLOC(float*, n);

	for(int i = 0; i < n; i++){
		r.perturbances[i] = ALLOC(float, num_params);
		
		for(int j = 0; j < num_params; j++)
			r.perturbances[i][j] = normal(0, r.std);
	}
	return r;
}

void rs_step(RS r){
	for(int i = 0; i < r.directions; i++)
		for(int j = 0; j < r.num_params; j++)
			r.perturbances[i][j] = normal(0, r.std);
}
