#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "lstm.h"


#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

typedef uint8_t bool;

/*
 * This file is for training an LSTM character-by-character on any text (ascii) file provided.
 */
 
//Calling convention: ../bin/char [load/new] [path_to_modelfile] [path_to_txt_file] [epochs] [learning_rate] [sample/train] [stateful]


static inline char int2char(int i){
	if(i==95) return '\n';
	return i+32;
}
static inline int char2int(char c){
	if(c=='\n') return 95;
	return c-32;
}
void bad_args(char *s){
	printf("bad argument '%s'\n", s);
	exit(1);
}

int main(int argc, char** argv){
	srand(time(NULL));
	setbuf(stdout, NULL);
	size_t ascii_range = 96; //96 useful characters in ascii: A-Z, a-z, 0-9, !@#$%...etc
	LSTM n = create_lstm(ascii_range, 40, ascii_range);
	CREATEONEHOT(x, ascii_range, 3);
	CREATEONEHOT(y, ascii_range, 5);
	lstm_forward(&n, x);
	n.cost(&n, y);
	lstm_backward(&n);
	/*
	if(argc < 2){ printf("must provide %d args.\n", 2); exit(1);}

	printf("received %d args: [", argc);
	for(int i = 0; i < argc; i++){
		printf("'%s'", argv[i]);
		if(i < argc-1)printf(", ");
		else printf("]\n");
	}
	bool newlstm;
	if(!strcmp(argv[1], "load")) newlstm = 0;
	else if(!strcmp(argv[1], "new")) newlstm = 1;
	else bad_args(argv[1]);


	printf("range: %lu\n", ascii_range);
	for(int i = 0; i < ascii_range; i++){
		printf("index %d has: '%c'\n", i, int2char(i));
	}*/


}
