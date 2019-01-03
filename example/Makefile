CC = gcc

BIN = ../bin

SRC_DIR = ../src
DAT_DIR = ../data

INCLUDE = -I../include
LIBS = -lm
CFLAGS = -O3

LSTM_SRC = $(SRC_DIR)/LSTM.c
MLP_SRC = $(SRC_DIR)/MLP.c
MNIST_SRC = $(SRC_DIR)/mnist.c

shakespeare:
	$(CC) $(CFLAGS) $(INCLUDE) $(LSTM_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)

mlp_mnist:
	$(CC) $(CFLAGS) $(INCLUDE) $(MNIST_SRC) $(MLP_SRC) ../examples/$@.c -o $(BIN)/$@ $(LIBS)
