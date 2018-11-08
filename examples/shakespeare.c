#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "RNN.h"

#define DROPOUT 0

/*
 * This is a demonstration of the network I am currently training on my home computer.
 * The network is trained character-by-character on Shakespeare's sonnets.
 * The end goal is that the network writes its own sonnets.
 * Currently it gets about two lines in and then repeats the same four or five words over and over.
 * At some point I will be writing a GPU kernel in CUDA or OpenCL so that this training can be done more quickly. 
 */

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()[]<>/'\"_\n ";

#if DROPOUT
char *modelfile = "../saves/rnn_shakespeare_3x600_DROPOUT.rnn";
#else
char *modelfile = "../saves/rnn_shakespeare_3x600.rnn";
#endif

char *datafile = "../shakespeare/complete_works.txt";
size_t datafilelen = 5447092;

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
	if(inpt == EOF) return EOF;
	return strlen(alphabet)-1;
}


int main(void){
	srand(time(NULL));
	setbuf(stdout, NULL);

	printf("Press ENTER to load %s (may take a while to load)\n", modelfile);
	RNN n;
	if(getchar() == 'n'){
		printf("creating network...\n");
		n = createRNN(strlen(alphabet), 600, 600, 600, strlen(alphabet));//loadRNNFromFile(modelfile);
	}else{
		printf("loading network from %s...\n", modelfile);
		n = loadRNNFromFile(modelfile);
	}
	
	Layer *current = n.input;
	while(current != NULL){
    if(!(current == n.input || current == n.output)){
#if DROPOUT
			current->dropout = 0.3;
#endif
//      current->squish = hypertan; //assigns this layer's squish function pointer to the tanh activation function
    }
		current = current->output_layer;
	}
	
	n.plasticity = 0.04; //I've found that the larger the network, the lower the initial learning rate should be.	

	int epochs = 1000;
  float previousepochavgcost = 4.5;
	for(int i = 0; i < epochs; i++){ //Run for a large number of epochs
		FILE *fp = fopen(datafile, "rb"); //This is the dataset
		if(!fp){
			printf("%s COULD NOT BE OPENED!\n", datafile);
			exit(1);
		}
	
		char input_character = ' '; //Start the network off with a character
		char label = label_from_char(fgetc(fp), alphabet); //Get the first letter from the dataset as a label

		float cost = 0;
		float linecost = 0;
		size_t linecount = 1;
		int count = 0;
		float lastavgcost = previousepochavgcost;
		size_t epochcount = 1;
		float epochcost = 0;
		do {
			//The below is all the code needed for training - the rest is just debug stuff.
			/****************************************************/
			float input_one_hot[strlen(alphabet)];
			make_one_hot(input_character, alphabet, input_one_hot);	
			setOneHotInput(&n, input_one_hot);
			
			float cost_local = step(&n, label);
			/****************************************************/

			cost += cost_local;
			linecost += cost_local;
			if(alphabet[label] == '\n') printf("\n");
			else if(alphabet[bestGuess(&n)] == alphabet[label]) printf("%c", alphabet[label]);
			else printf("_");
			input_character = alphabet[label];
			count++;
			linecount++;
			label = label_from_char(fgetc(fp), alphabet);
			if(count % 1000 == 0){
				epochcount += count;
				epochcost += cost;
			
				//Debug stuff
				float completion = 100 * (float)epochcount/datafilelen;
				float percentchange = 100 * ((lastavgcost / (cost/count)) - 1);
				float percentchange_epoch = 100 * ((previousepochavgcost / (epochcost/epochcount)) - 1);
				printf("\n\n****\nlatest cost: %f vs epoch avg cost:%f, epoch %5.2f%% completed.\n", cost/count, epochcost/epochcount, completion);
				printf("%5.2f%% improvement over last 1000 chars, %5.3f%% since last epoch.\n", percentchange, percentchange_epoch);

				if(epochcount % 15000 == 0 && percentchange_epoch > 0){
					printf("\n\n***************\nAUTOSAVING MODEL FILE!\n");
					saveRNNToFile(&n, modelfile);
					printf("***************\n\n");
				}

				Neuron *output_neuron = &n.output->neurons[bestGuess(&n)];
				printf("output neuron (%d) has gradient %f, dActivation %f, output %f, \'%c\'\n*****\n", bestGuess(&n), output_neuron->gradient, output_neuron->dActivation, output_neuron->activation, alphabet[bestGuess(&n)]);

				//Reset short-term cost statistic
				cost = 0;
				count = 0;
			}
		}
	
		while(label != EOF);
		fclose(fp);
		printf("Epoch completed, cost was %f vs previous cost of %f\n", cost/count, lastavgcost);
		saveRNNToFile(&n, modelfile); 
		previousepochavgcost = epochcost/epochcount;

		//Get a sample sonnet by feeding the network its own output, starting with a random letter.
		char input = alphabet[rand()%(strlen(alphabet)-1)];
		printf("Sample from input '%c':\n", input);
		for(int i = 0; i < 1000; i++){
			float input_vector[strlen(alphabet)];		
			make_one_hot(input, alphabet, input_vector);
			
			setOneHotInput(&n, input_vector);
			
			feedforward_recurrent(&n);
	
			input = alphabet[bestGuess(&n)];
			printf("%c", input);
		}
	}
}
