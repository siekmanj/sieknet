#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

/*
 * This is a demonstration of the network I am currently training on my home computer.
 * The network is trained character-by-character on Shakespeare's sonnets.
 * The end goal is that the network writes its own sonnets.
 * Currently it gets about two lines in and then repeats the same four or five words over and over.
 * At some point I will be writing a GPU kernel in CUDA or OpenCL so that this training can be done more quickly. 
 */

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()[]'\"\n ";


/*
 * Description: This is a function that uses an input character to create a one-hot input vector.
 * inpt: the character whose position in alphabet will be used to set a single element to 1.0 in dest.
 * alphabet: the list of all possible inputs and outputs of the network (needs to be the same length as input/output layer of network)
 * dest: the output float array.
 */
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

/*
 * Description: This function turns a character into a label using that character's position in an alphabet.
 * inpt: the character whose position in alphabet will be used to return a label.
 * alphabet: the list of all possible inputs and outputs of the network (needs to be the same length as input/output layer of network)
 */
int label_from_char(char inpt, const char *alphabet){
	for(int i = 0; i < strlen(alphabet); i++){
		if(alphabet[i] == inpt) return i;
	}
	return EOF;
}


int main(void){
	srand(time(NULL));
	setbuf(stdout, NULL);

	char* filename = "../saves/rnn_sonnets_600x900x500.rnn"; //This is the network file that will be loaded, and the one that will be saved to.

	printf("Press ENTER to load %s (may take a while to load)\n", filename);
	RNN n;
	if(getchar() == 'n'){
		printf("creating network...\n");
		n = createRNN(strlen(alphabet), 600, 900, 500, strlen(alphabet));//loadRNNFromFile(filename);
	}else{
		printf("loading network from %s...\n", filename);
		n = loadRNNFromFile(filename);
	}
	
	n.plasticity = 0.05; //I've found that the larger the network, the lower the initial learning rate should be.	

	int count = 0;
	int epochs = 1000;
	float previousepochavgcost = 1000000000;

	for(int i = 0; i < epochs; i++){ //Run for a large number of epochs
		FILE *fp = fopen("../shakespeare/sonnets.txt", "rb"); //This is the dataset
	
		char input_character = ' '; //Start the network off with a character
		char label = label_from_char(fgetc(fp), alphabet); //Get the first letter from the dataset as a label

		float cost = 0;
		float lastavgcost = 10000000;
		float epochcost = 0;
		float epochcount = 0;

		do {
			float input_one_hot[strlen(alphabet)];
			make_one_hot(input_character, alphabet, input_one_hot);	
			setOneHotInput(&n, input_one_hot);
			
			float cost_local = step(&n, label);

			cost += cost_local;
			if(alphabet[bestGuess(&n)] == alphabet[label]) printf("%c", alphabet[label]);
			else if(alphabet[label] == '\n') printf("\n");
			else printf("_");
			input_character = alphabet[label];
			count++;
		
			label = label_from_char(fgetc(fp), alphabet);

			if(count % 1000 == 0){
				epochcount += count;
				epochcost += cost;
			
				//Debug stuff
				printf("\n\n****\nlatest cost: %f vs previous cost: %f vs epoch avg cost:%f, epoch %5.2f%% completed.\n", cost/count, lastavgcost, epochcost/epochcount, 100 * (float)epochcount/102892.0);
				Neuron *output_neuron = &n.output->neurons[bestGuess(&n)];
				printf("output neuron (%d) has gradient %f, dActivation %f, output %f, \'%c\'\n*****\n", bestGuess(&n), output_neuron->activationGradient, output_neuron->dActivation, output_neuron->activation, alphabet[bestGuess(&n)]);

				lastavgcost = cost/count;
		
				//Reset short-term cost statistic
				cost = 0;
				count = 0;
			}
		}
	
		while(label != EOF);
		fclose(fp);

		printf("\n\n***********\nepoch %d concluded, avgcost: %f (vs previous %f).\n************\n\n", i, epochcost/epochcount, previousepochavgcost);

		//If the network did worse this epoch than the last, don't save the state and lower the learning rate.
		if(previousepochavgcost < epochcost/epochcount){
			printf("performance this epoch was worse than the one before. Plasticity next epoch will be %f.\n", n.plasticity * 1.0);
			n.plasticity *= 1.0;
		}else{
		  saveRNNToFile(&n, filename); 
    }
		previousepochavgcost = epochcost/epochcount;

		//Get a sample sonnet by feeding the network its own output, starting with a random letter.
		char input = alphabet[rand()%(strlen(alphabet)-1)];
		printf("Sample from input '%c':\n", input);
		for(int i = 0; i < 2000; i++){
			float input_vector[strlen(alphabet)];		
			make_one_hot(input, alphabet, input_vector);
			
			setOneHotInput(&n, input_vector);
			
			feedforward_recurrent(&n);
	
			input = alphabet[bestGuess(&n)];
			printf("%c", input);
		}
		if(i % 30 == 0 && i != 0) getchar(); //wait for user input to continue after 30 epochs
	}
}
