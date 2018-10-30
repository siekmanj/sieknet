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

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!-()[]<>/'\"_\n ";

char *modelfile = "../saves/rnn_shakespeare_3x350.rnn";
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
		n = createRNN(strlen(alphabet), 350, 350, 350, strlen(alphabet));//loadRNNFromFile(modelfile);
	}else{
		printf("loading network from %s...\n", modelfile);
		n = loadRNNFromFile(modelfile);
	}
	
	
	Layer *current = n.input;
	while(current != NULL){
    if(!(current == n.input || current == n.output)){
//      current->squish = hypertan; //assigns this layer's squish function pointer to the tanh activation function
//			current->dropout = 0.3;
    }
		current = current->output_layer;
	}
	
	n.plasticity = 0.0005; //I've found that the larger the network, the lower the initial learning rate should be.	

	int epochs = 1000;
	float previousepochavgcost = 100; 

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
		float linecount = 1;
		int count = 0;
		float lastavgcost = 10000000;
		printf("avgcost: %5.3f, epoch %d, completion %2.2f%%, current line:", cost/count, i, 100*(float)count/datafilelen);
		do {
			float input_one_hot[strlen(alphabet)];
			make_one_hot(input_character, alphabet, input_one_hot);	
			setOneHotInput(&n, input_one_hot);
			
			float cost_local = step(&n, label);

			cost += cost_local;
			linecost += cost_local;
			if(alphabet[label] == '\n'){
				linecost /= linecount;
				linecount = 0;
				printf("\r                                                                                                                                     ");
				printf("\ravgcost: %5.3f, linecost: %5.3f, epoch %d, completion %2.2f%%, current line:", cost/count, linecost, i, 100*(float)count/datafilelen);
				linecost = 0;
			}
			else if(alphabet[bestGuess(&n)] == alphabet[label]) printf("%c", alphabet[label]);
			else printf("_");
			input_character = alphabet[label];
			count++;
			linecount++;
			label = label_from_char(fgetc(fp), alphabet);
/*	
			if(count % 1000 == 0){
				epochcount += count;
				epochcost += cost;
			
				//Debug stuff
				printf("\n\n****\nlatest cost: %f vs previous cost: %f vs epoch avg cost:%f, epoch %5.2f%% completed.\n", cost/count, lastavgcost, epochcost/epochcount, 100 * (float)epochcount/5338134.0);
				Neuron *output_neuron = &n.output->neurons[bestGuess(&n)];
				printf("output neuron (%d) has gradient %f, dActivation %f, output %f, \'%c\'\n*****\n", bestGuess(&n), output_neuron->gradient, output_neuron->dActivation, output_neuron->activation, alphabet[bestGuess(&n)]);

				lastavgcost = cost/count;
		
				//Reset short-term cost statistic
				cost = 0;
				count = 0;
			}
*/
		}
	
		while(label != EOF);
		fclose(fp);
		printf("Epoch completed, cost was %f vs previous cost of %f\n", cost/count, lastavgcost);
		//If the network did worse this epoch than the last, don't save the state and lower the learning rate.
		if(lastavgcost < cost/count){
			printf("performance this epoch was worse than the one before. Plasticity next epoch will be %f.\n", n.plasticity * 0.97);
			n.plasticity *= .97;
		}else{
		  saveRNNToFile(&n, modelfile); 
		}
		lastavgcost = cost/count;

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
