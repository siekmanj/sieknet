CC = gcc
.PHONY: filesavetest mlp_mnist binary

MLP_INCLUDE = -I../mnist -I../networks/MLP
RNN_INCLUDE = -I../networks/RNN
MJ_INCLUDE = -I$(HOME)/.mujoco/mjpro150/include/ -L$(HOME)/.mujoco/mjpro150/bin -mavx
LSTM_INCLUDE = -I../networks/LSTM

C_FLAGS = -O0 
#C_FLAGS = -O3

lstm_test:
	gcc $(C_FLAGS) $(LSTM_INCLUDE) $(MLP_INCLUDE) ../networks/LSTM/LSTM.c ../networks/MLP/MLP.c ../examples/lstm_test.c -o $@.out -lm
sanity:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/sanitycheck.c -o $@.out -lm
walking:
	gcc $(MJ_INCLUDE) ../examples/walking.c -lmujoco150 -lGL -lglew ~/.mujoco/mjpro150/bin/libglfw.so.3 -o $@.out
genetic:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/genetic.c -o $@.out -lm
recurrent_demo:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/recurrent_demo.c -o $@.out -lm
sequence:
	gcc $(C_FLAGS) $(RNN_INCLUDE) $(MLP_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../examples/sequence.c -o $@.out -lm
shakespeare:
	gcc $(C_FLAGS) $(LSTM_INCLUDE) ../networks/LSTM/LSTM.c ../examples/shakespeare.c -o $@.out -lm
character:
	gcc $(C_FLAGS) $(MLP_INCLUDE) $(RNN_INCLUDE) $(LSTM_INCLUDE) ../networks/MLP/MLP.c ../networks/RNN/RNN.c ../networks/LSTM/LSTM.c ../examples/character.c -o $@.out -lm
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
