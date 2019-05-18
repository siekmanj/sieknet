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

	if(seed)
		r.params = seed;
	else
		r.params = calloc(num_params, sizeof(float));

	r.update = calloc(num_params, sizeof(float));

	r.deltas = ALLOC(Delta*, n);

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
			qsort(r.deltas, r.directions, sizeof(Delta*), rs_comparator);
			float mean = 0;
			float std  = 0;

			int b = r.directions - (int)((r.cutoff)*r.directions);

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
  r.optim.learning_rate = r.step_size;
  r.optim.step(r.optim);
	for(int i = 0; i < r.directions; i++)
		for(int j = 0; j < r.num_params; j++)
			r.deltas[i]->p[j] = normal(0, r.std);

}
