#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

void make_one_hot(char inpt, const char *alphabet, float *dest){
	for(int i = 0; i < strlen(alphabet); i++){
		if(inpt == alphabet[i]){
			dest[i] = 1.0;
		}
		else{
			dest[i] = 0.0;
		}
	}
	return;
}

int label_from_char(char inpt, const char *alphabet){
	for(int i = 0; i < strlen(alphabet); i++){
		if(alphabet[i] == inpt) return i;
	}
	printf("ERRRRRRR: %d (%c) not in alphabet.\n", inpt, inpt);
	while(1);
	return -1;
}

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()'\"\n ";

int main(void){
	RNN n;

	if(getchar() == 'n'){
		printf("creating new network...\n");
		n	= createRNN(strlen(alphabet), 100, 200, 100, strlen(alphabet));
	}else{
		printf("loading network from file...\n");
		RNN n = loadRNNFromFile("../saves/rnn_sonnets.rnn");
	}

	n.plasticity = 0.2;	

	int count = 0;
	int debug = 0;
	int epochs = 10;

	for(int i = 0; i < epochs; i++){
		FILE *fp = fopen("../shakespeare/sonnets.txt", "rb");
		char c = fgetc(fp);
		
		float cost = 0;
		float lastavgcost = 10000000;

		while(c != EOF){
			float input_one_hot[strlen(alphabet)];
			make_one_hot(c, alphabet, input_one_hot);	
			setOneHotInput(&n, input_one_hot);
			
			char label = label_from_char(fgetc(fp), alphabet); 
			
			cost += step(&n, label);
	
			if(count % 5000 == 0){
				debug ^= 1;	
				n.plasticity *= 0.9;
				if(debug){
					printf("\n\n************\navgcost: %f, plasticity: %f\n************\n\n", cost/count, n.plasticity);
					if(lastavgcost > cost/count){		
						saveRNNToFile(&n, "../saves/rnn_sonnets.rnn"); 
						lastavgcost = cost/count;
						printf("trying to beat cost of %f\n", lastavgcost);
					}else{
						printf("\n\n************\nPROGRESS LOST!!! %f vs %f\n************\n\n", cost/count, lastavgcost);
					}
				}
			}
			if(debug){
				printf("%c", alphabet[bestGuess(&n)]); 
			}
			c = label;
			count++;
		}
		fclose(fp);
		printf("\n\n***********\nepoch %d concluded, avgcost: %f.\n************\n\n", i, cost/count);
		getchar();
	}
}
