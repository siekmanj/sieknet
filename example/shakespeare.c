#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define UNROLL_LENGTH 45
#include "LSTM.h"


#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

/*
 * This LSTM is trained character-by-character on Shakespeare's complete works.
 * The end goal is that the network writes its own plays.
 */

char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.;:?!'\"[]{}<>/-_*&|\\\n ";

char *modelfile = "../model/shakespeare.lstm";

char *datafile = "../data/shakespeare/complete_works.txt";


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
	LSTM n;
	if(getchar() == 'n'){
		printf("creating network...\n");
		n = createLSTM(strlen(alphabet), 300, strlen(alphabet));//loadLSTMFromFile(modelfile);
	}else{
		printf("loading network from %s...\nenter 's' to sample.", modelfile);
		n = loadLSTMFromFile(modelfile);
		if(getchar() == 's'){
				int in = rand() % strlen(alphabet) - 1;
				printf("small sample from char '%c'\n", alphabet[in]);
				for(int k = 0; k < 500; k++){
					CREATEONEHOT(input, strlen(alphabet), in);
					forward(&n, input);
					int guess = 0;
					for(int k = 0; k < strlen(alphabet); k++) if(n.tail->output[k] > n.tail->output[guess]) guess = k;
					in = guess;
					printf("%c", alphabet[guess]);
			}
			exit(0);
		}
	}
	
	n.plasticity = 0.005; //I've found that the larger the network, the lower the initial learning rate should be.	

	int epochs = 1000;
  float previousepochavgcost = 4.5;
	for(int i = 0; i < epochs; i++){ //Run for a large number of epochs
		printf("beginning epoch %d\n", i);
		wipe(&n);
		FILE *fp = fopen(datafile, "rb"); //This is the dataset
		if(!fp){
			printf("%s COULD NOT BE OPENED!\n", datafile);
			exit(1);
		}
	
		int input_character = label_from_char(' ', alphabet); //Start the network off with a character
		int label = label_from_char(fgetc(fp), alphabet); //Get the first letter from the dataset as a label

		float cost = 0;
		float lastcost = 0;
		int count = 0;
		float lastavgcost = previousepochavgcost;
		size_t epochcount = 1;
		size_t datafilelen = 5447092;
		float epochcost = 0;
		do {
			//The below is all the code needed for training - the rest is just debug stuff.
			/****************************************************/
			CREATEONEHOT(input_one_hot, strlen(alphabet), input_character);
			CREATEONEHOT(expected, strlen(alphabet), label);

			forward(&n, input_one_hot);
			float cost_local = quadratic_cost(&n, expected);
			backward(&n);
			
			/****************************************************/
			if(isnan(cost_local)) { printf("COST NAN! STOPPING...\n"); exit(1); }

			cost += cost_local;

			int guess = 0;
			for(int k = 0; k < strlen(alphabet); k++) if(n.tail->output[k] > n.tail->output[guess]) guess = k;

			if(alphabet[label] == '\n') printf("\n");
			else if(alphabet[guess] == alphabet[label]) printf("%c", alphabet[label]);
			else printf("_");

			input_character = label;
			count++;
			label = label_from_char(fgetc(fp), alphabet);
			
			if(count % 10000 == 0){
				epochcount += count;
				epochcost += cost;
			
				//Debug stuff
				float completion = 100 * (float)epochcount/datafilelen;
				if(lastcost == 0) lastcost = cost/count;
				lastcost = (lastcost * 10 + cost/count)/11.0;
				printf("\n\n****\nlatest cost: %6.5f (avg %6.5f) vs epoch avg cost:%f, epoch (%d) %5.2f%% completed.\n", cost/count, lastcost, epochcost/epochcount, i, completion);

				if(cost/count < epochcost/epochcount){
					printf("\nAUTOSAVING MODEL FILE!\n");
					saveLSTMToFile(&n, modelfile);
				}else{
					printf("\nPERFORMANCE WORSE THAN AVERAGE, NOT SAVING MODELFILE\n");
				}
				printf("****\n\n");

				//Reset short-term cost statistic
				cost = 0;
				count = 0;
				sleep(1.0);
			}
		}
	
		while(label != EOF);
		fclose(fp);
		printf("Epoch completed, cost was %f vs previous cost of %f\n", cost/count, lastavgcost);
//		saveLSTMToFile(&n, modelfile); 
		previousepochavgcost = epochcost/epochcount;
//		getchar();

		//Get a sample sonnet by feeding the network its own output, starting with a random letter.
		/*
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
		*/
	}
}
