#include <conf.h>
#include <rs.h>
#include <math.h>

#ifdef SIEKNET_USE_OMP
#include <omp.h>
#endif

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return mean + norm * std;
}

RS create_rs(float (*R)(const float *, size_t), float *seed, size_t num_params, size_t n){
	RS r;
	if(!seed){
		printf("Error: parameter vector cannot be null!\n");
		exit(1);
	}
	r.std = 0.0075;
	r.step_size = 0.05;
	r.directions = n;
	r.num_params = num_params;
	r.algo = V1;
	r.cutoff = 0.0;
  r.num_threads = 1;

  r.f = R;

  r.params = seed;
	r.update = calloc(num_params, sizeof(float));
	r.deltas = calloc(n, sizeof(Delta*));
	r.optim = cpu_init_SGD(r.params, r.update, r.num_params);

	for(int i = 0; i < n; i++){
		r.deltas[i] = ALLOC(Delta, 1);
		r.deltas[i]->p = ALLOC(float, num_params);
		
		for(int j = 0; j < num_params; j++)
			r.deltas[i]->p[j] = normal(0, r.std);
	}
	return r;
}

static float max(const float a, const float b){
	return a > b ? a : b;
}

static int rs_comparator(const void *one, const void *two){
	Delta *a = *(Delta**)one;
	Delta *b = *(Delta**)two;

	float max_a = max(a->r_pos, a->r_neg);
	float max_b = max(b->r_pos, b->r_neg);

	if(max_a < max_b)
		return 1;
	if(max_a > max_b)
		return -1;
	else 
		return 0;
}

void rs_step(RS r){

  /* Allocate memory for parameter vector for each thread */
  float **thetas = ALLOC(float*, r.num_threads);
  for(int i = 0; i < r.num_threads; i++)
    thetas[i] = ALLOC(float, r.num_params);

  /* Do rollouts */
  #ifdef _OPENMP
  omp_set_num_threads(r.num_threads);
  #pragma omp parallel for default(none) shared(r, thetas)
  #endif
  for(int i = 0; i < r.directions; i++){
    #ifdef _OPENMP
    size_t thread = omp_get_thread_num();
    #else
    size_t thread = 0;
    #endif

    /* Positive delta rollouts */
    for(int j = 0; j < r.num_params; j++)
      thetas[thread][j] = r.params[j] + r.deltas[i]->p[j];

    r.deltas[i]->r_pos = r.f(thetas[thread], r.num_params);
    
    /* Negative delta rollouts */
    for(int j = 0; j < r.num_params; j++)
      thetas[thread][j] = r.params[j] - r.deltas[i]->p[j];

    r.deltas[i]->r_neg = r.f(thetas[thread], r.num_params);
  }
  //*(r.samples) += samples;

  for(int i = 0; i < r.num_threads; i++)
    free(thetas[i]);
  free(thetas);

	switch(r.algo){
		case BASIC:
		{
			for(int i = 0; i < r.directions; i++){
				float weight = -(r.step_size / r.directions) * (r.deltas[i]->r_pos - r.deltas[i]->r_neg);
				for(int j = 0; j < r.num_params; j++)
					r.update[j] += weight * r.deltas[i]->p[j];
			}
		}
		break;
		case V1:
		{
      /* Sort all noise vectors by performance */
			qsort(r.deltas, r.directions, sizeof(Delta*), rs_comparator);

      /* Use only top b noise vectors when calculating update */
			int b = r.directions - (int)((r.cutoff)*r.directions);

      /* Mean and standard deviation of reward calculation */
			float mean = 0;
			float std  = 0;

      for(int i = 0; i < b; i++){
        mean += r.deltas[i]->r_pos + r.deltas[i]->r_neg;
      }
      mean /= 2 * b;

      for(int i = 0; i < b; i++){
        float x = r.deltas[i]->r_pos;
        std += (x - mean) * (x - mean);
        x = r.deltas[i]->r_neg;
        std += (x - mean) * (x - mean);
      }
      std = sqrt(std/(2 * b));

      /* Sum up all the weighted noise vectors to get update */
      float weight = -1 / (b * std);
      for(int i = 0; i < b; i++){
        for(int j = 0; j < r.num_params; j++){
 					float reward = (r.deltas[i]->r_pos - r.deltas[i]->r_neg);
					float d = r.deltas[i]->p[j] / r.std;
					r.update[j] += weight * reward * d;
				}
			}

		}
		break;
	}
  /* Update the policy's parameters */
  r.optim.learning_rate = r.step_size;
  r.optim.step(r.optim);

  /* Generate deltas for next step */
  #ifdef _OPENMP
  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel for default(none) shared(r)
  #endif
	for(int i = 0; i < r.directions; i++)
		for(int j = 0; j < r.num_params; j++)
			r.deltas[i]->p[j] = normal(0, r.std);

}
