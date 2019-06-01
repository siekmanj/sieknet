#include <string.h>
#include <conf.h>
#include <env.h>
#include <rs.h>
#include <math.h>

#ifdef SIEKNET_USE_OMP
#include <omp.h>
#endif

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(1e-12, 1);
	float u2 = uniform(1e-12, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return mean + norm * std;
}

RS create_rs(float (*R)(const float *, size_t, Normalizer*), float *seed, size_t num_params, size_t n){
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
	r.top_b = 0.0;
  r.num_threads = 1;

  r.f = R;

  r.params = seed;
	r.update = calloc(num_params, sizeof(float));
	r.deltas = calloc(n, sizeof(Delta*));
#ifndef SIEKNET_USE_GPU
	r.optim = cpu_init_SGD(r.params, r.update, r.num_params);
#else
  printf("ERROR: create_rs(): random search currently not supported on GPU.\n");
  exit(1);
#endif
  r.normalizer = NULL;

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

  Normalizer **normalizers = NULL;
  if(r.normalizer){

    if(r.algo == V2 || r.algo == V2_t)
      r.normalizer->update = 1;
    else
      r.normalizer->update = 0;

    normalizers = ALLOC(Normalizer*, r.num_threads);
    for(int i = 0; i < r.num_threads; i++)
      normalizers[i] = copy_normalizer(r.normalizer);
    
    memset(r.normalizer->mean, '\0', sizeof(float)*r.normalizer->dimension);
    memset(r.normalizer->mean_diff, '\0', sizeof(float)*r.normalizer->dimension);
  }

  /* Do rollouts */
  #ifdef _OPENMP
  omp_set_num_threads(r.num_threads);
  #pragma omp parallel for default(none) shared(r, thetas, normalizers)
  #endif
  for(int i = 0; i < r.directions; i++){
    #ifdef _OPENMP
    size_t thread = omp_get_thread_num();
    #else
    size_t thread = 0;
    #endif
    
    Normalizer *norm = NULL;
    if(r.normalizer)
      norm = normalizers[thread];

    /* Positive delta rollouts */
    for(int j = 0; j < r.num_params; j++)
      thetas[thread][j] = r.params[j] + r.deltas[i]->p[j];

    r.deltas[i]->r_pos = r.f(thetas[thread], r.num_params, norm);
    
    /* Negative delta rollouts */
    for(int j = 0; j < r.num_params; j++)
      thetas[thread][j] = r.params[j] - r.deltas[i]->p[j];

    r.deltas[i]->r_neg = r.f(thetas[thread], r.num_params, norm);
  }

  size_t steps_before = 0;
  if(r.normalizer)
	  steps_before = r.normalizer->num_steps;
  for(int i = 0; i < r.num_threads; i++){
    if(r.normalizer){
      for(int j = 0; j < r.normalizer->dimension; j++){
        r.normalizer->mean[j] += normalizers[i]->mean[j] / r.num_threads;
        r.normalizer->mean_diff[j] += normalizers[i]->mean_diff[j] / r.num_threads;
      }
      r.normalizer->num_steps += normalizers[i]->num_steps - steps_before;
      dealloc_normalizer(normalizers[i]);
    }
    free(thetas[i]);
  }
  if(r.normalizer)
    free(normalizers);

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
    case V2:
    {
      if(!r.normalizer){
        printf("ERROR: rs_step(): normalizer not initialized.\n");
        exit(1);
      }
    } /* Intentional fall-through */
    case V1:
    {
      /* Consider all directions */
      int b = r.directions; 

      /* Mean and standard deviation of reward calculation */
			float mean = 0;
			float std  = 0;

      for(int i = 0; i < b; i++){
        mean += r.deltas[i]->r_pos + r.deltas[i]->r_neg;
#ifdef SIEKNET_DEBUG
        if(!isfinite(mean)){
          printf("WARNING: rs_step(): got non-finite mean while calculating reward stats, from %f or %f\n", r.deltas[i]->r_pos, r.deltas[i]->r_neg);
          exit(1);
        }
#endif
      }
      mean /= 2 * b;

      for(int i = 0; i < b; i++){
        float x = r.deltas[i]->r_pos;
        std += (x - mean) * (x - mean);
        x = r.deltas[i]->r_neg;
        std += (x - mean) * (x - mean);
#ifdef SIEKNET_DEBUG
				if(!isfinite(std)){
          printf("WARNING: rs_step(): got non-finite std during sum from either: %f, %f, or %f\n", mean, r.deltas[i]->r_pos, r.deltas[i]->r_neg);
          exit(1);
				}
#endif
      }
#ifdef SIEKNET_DEBUG
      if(!isfinite(sqrt(std/(2 * b)))){
        printf("WARNING: rs_step(): got non-finite standard deviation of reward form sqrt(%f / (2 * %d))\n", std, b);
        exit(1);
      }
#endif
      std = sqrt(std/(2 * b));

      /* Sum up all the weighted noise vectors to get update */
      float weight = -1 / (b * std);
      for(int i = 0; i < b; i++){
        for(int j = 0; j < r.num_params; j++){
          float reward = (r.deltas[i]->r_pos - r.deltas[i]->r_neg);
          float d = r.deltas[i]->p[j] / r.std;
          r.update[j] += weight * reward * d;
#ifdef SIEKNET_DEBUG
          if(!isfinite(r.update[j])){
            printf("WARNING: rs_step(): got non-finite gradient estimate from %f * %f * %f\n", weight, reward, d);
            exit(1);
          }
#endif
				}
			}
    }
    break;
    case V2_t:
    {
      if(!r.normalizer){
        printf("ERROR: rs_step(): normalizer not initialized.\n");
        exit(1);
      }
    }
		case V1_t:
		{
      /* Sort all noise vectors by performance */
			qsort(r.deltas, r.directions, sizeof(Delta*), rs_comparator);

      /* Use only top b noise vectors when calculating update */
			int b = (int)((r.top_b)*r.directions);

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
		for(int j = 0; j < r.num_params; j++){
			r.deltas[i]->p[j] = normal(0, r.std);
			if(!isfinite(r.deltas[i]->p[j])){
				printf("ERROR: rs_step(): got non-finite value whil generating noise vector for direction %d, param %d: %f\n", i, j, r.deltas[i]->p[j]);
				exit(1);
			}
		}

}
