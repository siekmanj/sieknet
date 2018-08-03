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
	addLayer(&n, 28*28); //input layer
	addLayer(&n, 15);
	addLayer(&n, 10); //output layer
	saveToFile(&n, "test");
}


int main(){
	srand(time(NULL));
	//binarySolver();
	//mnist();
	savetest();



}
