#include <stdio.h>
#include <math.h>
#include <mlp.h>
#include <string.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}
/*
 * This is an example problem that demonstrates a very simple use of a multilayer perceptron.
 * The network is trained to convert a 4-bit binary string into a decimal number.
 */

void cmp_vec(float *a, float *b, size_t len){
	for(int i = 0; i < len; i++){
		if(a[i] != b[i]){
			printf("WARNING: a[%d]: %5.4f != b[%d]: %5.4f\n", i, a[i], i, b[i]);
		}else
			printf("a[%d]: %5.4f == b[%d]: %5.4f\n", i, a[i], i, b[i]);
	}
}

float elem_sigm(float x){
	return 1 / (1 + exp(-x));
}

int main(){
	srand(time(NULL));
	/*
	MLP n = createMLP(2, 2, 2); //Create a network with a 4-neuron input layer, 8-neuron hidden layer, and 16-neuron output layer.
	n.learning_rate = 1.0;

	float x[] = {0.2, 0.3};
	float y[] = {1.0, 0.0};

	n.layers[0].neurons[0].weights[0] = 0.9;
	n.layers[0].neurons[0].weights[1] = 0.8;
	n.layers[0].neurons[1].weights[0] = 0.6;
	n.layers[0].neurons[1].weights[1] = 0.1;
	*n.layers[0].neurons[0].bias = 0.3;
	*n.layers[0].neurons[1].bias = 0.4;

	n.layers[1].neurons[0].weights[0] = 0.95;
	n.layers[1].neurons[0].weights[1] = 0.1;
	n.layers[1].neurons[1].weights[0] = 0.35;
	n.layers[1].neurons[1].weights[1] = 0.2;
	*n.layers[1].neurons[0].bias = 0.7;
	*n.layers[1].neurons[1].bias = 0.1;


	float l1_expected_outputs[] = {elem_sigm(0.72), elem_sigm(0.55)};
	float l2_expected_outputs[] = {elem_sigm(1.40239), elem_sigm(0.46223957)};

	mlp_forward(&n, x);

	printf("comparing l1 outs to expected:\n");
	cmp_vec(n.layers[0].output, l1_expected_outputs, 2);
	printf("comparing l2 outs to expected:\n");
	cmp_vec(n.layers[1].output, l2_expected_outputs, 2);

	float expected_costs[] = {y[0] - l2_expected_outputs[0], y[1] - l2_expected_outputs[1]};
	float cost = n.cost(&n, y);
	printf("comparing expected cost grads:\n");
	cmp_vec(n.cost_gradient, expected_costs, n.output_dimension);

	printf("BEFORE\n");
	for(int i = 0; i < n.num_params; i++)
		printf("%5.4f\n", n.params[i]);
	mlp_backward(&n);
	printf("AFTER\n");
	for(int i = 0; i < n.num_params; i++)
		printf("%5.4f\n", n.params[i]);
	printf("weight should be 0.35 - 0.0978: %5.4f\n", n.layers[1].neurons[1].weights[0]);

	getchar();
	*/
	MLP n = createMLP(4, 32, 16);
	//MLP n = load_mlp("../model/bin.mlp");
	n.learning_rate = 0.1;
	n.layers[0].logistic = relu;
	float avg_cost;
	//n.cost = cross_entropy_cost;
	for(int i = 0; i < 100000; i++){ //Run the network for 80000....00 examples
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float x[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		CREATEONEHOT(y, 16, (int)ans);

		mlp_forward(&n, x);
		float cost = n.cost(&n, y);
		mlp_backward(&n);

		avg_cost += cost;

		//Debug stuff
		if(!(i % 1000)){
			printf("CURRENTLY ON EXAMPLE %d\n", i);
			printf("Label %2d, guess %2d, Cost: %5.3f, avg: %5.3f\n\n(ENTER to continue, CTRL+C to quit)\n", (int)ans, n.guess, cost, avg_cost/i);
			getchar();
		}	
	}
	save_mlp(&n, "../model/bin.mlp");
	dealloc_network(&n);
}
