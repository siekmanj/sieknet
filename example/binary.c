/* Jonah Siekmann
 * 1/20/2019
 * This is a toy problem - training an mlp to convert a 4-bit binary string to a decimal number between 0 and 15.
 */

#include <mlp.h>
#include <optimizer.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

int main(){
	srand(time(NULL));

	MLP n = create_mlp(4, 16);
  //Momentum o = create_optimizer(Momentum, n);

	//n.layers[0].logistic = relu;
	float avg_cost;
	for(int i = 0; i < 100000; i++){ //Run the network for a while
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float x[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		CREATEONEHOT(y, 16, (int)ans);

		mlp_forward(&n, x);
		float cost = mlp_cost(&n, y);
		mlp_backward(&n);

		//o.step(o);

		avg_cost += cost;

		//Debug stuff
		if(!(i % 1000)){
			printf("CURRENTLY ON EXAMPLE %d\n", i);
			printf("Label %2d, guess %2lu, Cost: %5.3f, avg: %5.3f\n\n(ENTER to continue, CTRL+C to quit)\n", (int)ans, n.guess, cost, avg_cost/i);
			getchar();
		}	
	}
	save_mlp(&n, "../model/binary.mlp");
	dealloc_mlp(&n);
}
