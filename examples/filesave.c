/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <MLP.h>
#include <mnist.h>

const int RANGE = 7;

void savetest(){
	MLP n = initMLP();
	addLayer(&n, 28*27); //input layer
	addLayer(&n, 3);
	addLayer(&n, 4); //output layer
	saveMLPToFile(&n, "../saves/thisatest.mlp");

	MLP recovered = loadMLPFromFile("../saves/thisatest.mlp");
}


int main(){
	srand(time(NULL));
	//binarySolver();
	//mnist();
	savetest();



}
