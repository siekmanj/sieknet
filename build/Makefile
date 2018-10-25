CC = gcc
.PHONY: filesavetest mlp_mnist binary

MLP_INCLUDE = -I../mnist -I../networks/MLP
RNN_INCLUDE = -I../networks/RNN
C_FLAGS = -O3
genetic:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/recurrent_demo.c -o $@.out -lm
recurrent_demo:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/recurrent_demo.c -o $@.out -lm
sequence:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/sequence.c -o $@.out -lm
shakespeare:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/shakespeare.c -o $@.out -lm
character:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/character.c -o $@.out -lm
filesave:
	gcc $(C_FLAGS) $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/filesave.c -o $@.out -lm
fileload:
	gcc $(C_FLAGS) $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/fileload.c -o $@.out -lm
mlp_mnist:
	gcc $(C_FLAGS) $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/mlp_mnist.c -o $@.out -lm
binary:
	gcc $(C_FLAGS) $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/binary.c -o $@.out -lm
clean:
	rm *.out
