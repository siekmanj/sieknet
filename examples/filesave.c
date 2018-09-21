/* 
 * Jonah Siekmann
 * 7/24/2018
 */

#include <stdio.h>
#include <MLP.h>

/*
 * This is a simple example of how to use the provided saveMLPToFile and loadMLPFromFile functions.
 * The saveRNNToFile and loadRNNFromFile functions work identically.
 */

int main(void){
	MLP n = createMLP(1, 2, 3);
	saveMLPToFile(&n, "../saves/thisatest.mlp");
	MLP recovered = loadMLPFromFile("../saves/thisatest.mlp");
}

