CC = gcc
.PHONY: filesavetest mlp_mnist binary

MLP_INCLUDE = -I../mnist -I../networks/MLP

filesave:
	gcc $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/filesave.c -o $@.out -lm
fileload:
	gcc $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/fileload.c -o $@.out -lm
mlp_mnist:
	gcc $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/mlp_mnist.c -o $@.out -lm
binary:
	gcc $(MLP_INCLUDE) ../mnist/mnist.c ../networks/MLP/MLP.c ../examples/binary.c -o $@.out -lm
clean:
	rm *.out
