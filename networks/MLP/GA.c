/*
 * This is a file that stores functions for training with a genetic algorithm (GA)
 * instead of backpropagation.
 */

#include "GA.h"

/* Description: Calculates the gradients of the output layer with respect to the activation of each neuron.
 *              This function is intended to be used when learning via genetic algorithm.
 * output_layer: the last layer in the network.
 */
static void gradients_wrt_outputs(Layer *output_layer){
	for(int i = 0; i < output_layer->size; i++){
		output_layer->neurons[i].gradient = -1 * output_layer->neurons[i].dActivation;
	}
	propagate_gradients(output_layer);
}

/* Description: Calculates gradients with respect to output layer activations, then uses those gradients
 *              to decide how much to randomly change a parameter by. This was described
 *              in a 2017 Uber paper and this is my implementation of it.
 * n: The pointer to the network.
 * mutation_rate: The proportion of neurons which will mutate
 */
static void mutate(MLP *n, float mutation_rate){
	Layer *current = n->output;
	gradients_wrt_outputs(current); //Calculate gradients with respect to outputs of output layer for every neuron in network.

	while(current->input_layer != NULL){
		Layer *input_layer = current->input_layer;

		for(int i = 0; i < current->size; i++){
			Neuron *neuron = &current->neurons[i];
//			printf("Considering mutating %p weights\n.", neuron);
			float dActivation = neuron->dActivation;
			float gradient = neuron->gradient;

			for(int j = 0; j < input_layer->size; j++){
				if(mutation_rate > (rand()%1000)/1000.0){
					Neuron *input_neuron = &input_layer->neurons[j];
					float a = input_neuron->activation;
					float weight_gradient = 1/exp(a * dActivation * gradient);
//					printf("Mutating! Weight %d of %p incrementing by %f\n", j, neuron, weight_gradient * n->plasticity);
					if(rand()&1) weight_gradient *= -1;
					neuron->weights[j] += weight_gradient * n->plasticity;
				}
			}
			if(mutation_rate > (rand()%1000)/1000.0){
				float bias_gradient = 1/exp(dActivation * gradient);
				if(rand()&1) bias_gradient *= -1;
				neuron->bias += bias_gradient * n->plasticity;
//				printf("Mutating! Bias of %p incrementing by %f\n", neuron, dActivation * gradient * n->plasticity);
			}
		}
		current = input_layer;
	}
}

/*
 * This function copies the dimensions and parameters of an existing MLP to
 * a new MLP.

 * n: the network to be copied.
 */
MLP copy_mlp(MLP *n){
	MLP ret = initMLP();
	Layer *current = n->input;

	while(current != NULL){
		addLayer(&ret, current->size);
		ret.output->squish = current->squish;
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
				for(int j = 0; j < current->input_layer->size; j++){
					ret.output->neurons[i].weights[j] = current->neurons[i].weights[j];
				}
				ret.output->neurons[i].bias = current->neurons[i].bias;
			}
		}
		current = current->output_layer;
	}
	return ret;
}

/*
 * This function finds the matching neuron in two networks with identical sizes.
 * This is necessary since layers are stored in a linked list.
 
 * n: the network to be searched
 * layer_idx: the nth layer of the network.
 * neuron_idx: the nth neuron of the layer
 */
static Neuron *neuron_lookup(MLP *n, int layer_idx, int neuron_idx){
	Layer *current = n->input;
	for(int i = 0; i < layer_idx; i++){
		current = current->output_layer;
	}
	return &current->neurons[neuron_idx];
}

/*
 * Compares the parameters of two networks to determine how similar they are.
 * Returns a number between 0 and 1.

 * a, b: the two networks to be compared.
 */
float similarity(MLP *a, MLP *b){
	int total = 0;
	int identical = 0;
	int layeridx = 0;
	Layer *current = a->input;
	while(current != NULL){
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
				Neuron *comparator = neuron_lookup(b, layeridx, i);
				for(int j = 0; j < current->input_layer->size; j++){
					if(comparator->weights[j] == current->neurons[i].weights[j]){
							identical++;
					}
					total++;
				}
			}	
		}
		current = current->output_layer;
		layeridx++;
	}
	return ((float)identical)/total;
}

/*
 * Creates a child MLP from two parents, randomly choosing parameters from each.
 * 
 * partner1, partner2: the two parents which will be used as source material for the child.
 */
MLP crossbreed(MLP *partner1, MLP *partner2){
//	printf("copying...\n");
	MLP ret = copy_mlp(partner1);
//	printf("copy done,\n");
//	print_layers(&ret);
	Layer *current = ret.input;
	int layer_idx = 0;
	while(current != NULL){
//		printf("Starting layer %d, pointer: %p, inpt: %p\n", layer_idx, current, current->input_layer);
		if(current->input_layer != NULL){
			for(int i = 0; i < current->size; i++){
				Neuron *partner_neuron = neuron_lookup(partner2, layer_idx, i);
				for(int j = 0; j < current->input_layer->size; j++){
//					printf("	Doing layer %d, neuron %d, weight %d\n", layer_idx, i, j);
					if(!(rand()%2)) current->neurons[i].weights[j] = partner_neuron->weights[j];
				}
				if(!(rand()%2)) current->neurons[i].bias = partner_neuron->bias;
//				printf("	Done with neuron %d\n", i);
			}
		}
//		printf("	Done with layer %d\n", layer_idx);
		layer_idx++;
		current = current->output_layer;
//		printf("	Moving on to layer %d\n", layer_idx);
	}
//	printf("Exiting crossbreed\n");
	return ret;
}

/*
 * Initializes a pool of MLP's from a single MLP.
 * The MLP passed in needs to be initialized with the desired
 * number of layers and dimensions. Pool->size must be initialized.

 * n: A network struct that the pool will match dimensionally.
 * pool: A pool struct which will be written to.
 */
void pool_from_mlp(MLP *n, Pool *pool){
	pool->pool = (MLP *)malloc(sizeof(MLP) * pool->size);
	for(int i = 0; i < pool->size; i++){
		pool->pool[i] = copy_mlp(n);
		pool->pool[i].plasticity = pool->plasticity;
		mutate(pool->pool[i], pool->mutation_rate);
	}
	mutate(pool);
}

int fitness_comp(const void *mlp1, const void *mlp2){
	MLP f1 = *(MLP*)mlp1;
	MLP f2 = *(MLP*)mlp2;
	if(f1.performance < f2.performance) return 1;
	if(f1.performance > f2.performance) return -1;
	else return 0;
}

void evolve(Pool *p){
	qsort(p->pool, p->size, sizeof(MLP), fitness_comp); //Order pool by highest fitness
	for(int j = p->size/2; j < p->size; j++){
		dealloc_network(&pool[j]);
		MLP *parent1 = &pool[rand()%(p->size/2)];
		MLP *parent2 = &pool[rand()%(p->size/2)];
		p->pool[j] = crossbreed(parent1, parent2);
		mutate(pool[j].output, LEARNING_RATE, MUTATION_RATE);
	}
}




