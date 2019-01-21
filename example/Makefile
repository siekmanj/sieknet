CC=gcc
DIRS=../bin ../model ../data ../log

BIN=../bin

SRC_DIR=../src
DAT_DIR=../data

INCLUDE=-I../include
LIBS=-lm
CFLAGS=-O3

LSTM_SRC=$(SRC_DIR)/lstm.c
MLP_SRC=$(SRC_DIR)/mlp.c
MNIST_SRC=$(SRC_DIR)/mnist.c

char:
	$(CC) $(CFLAGS) $(INCLUDE) $(LSTM_SRC) $(MLP_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)
shakespeare:
	$(CC) $(CFLAGS) $(INCLUDE) $(LSTM_SRC) $(MLP_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)
mlp_mnist:
	$(CC) $(CFLAGS) $(INCLUDE) $(MNIST_SRC) $(MLP_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)
binary:
	$(CC) $(CFLAGS) $(INCLUDE) $(MLP_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)

sequence:
	$(CC) $(CFLAGS) $(INCLUDE) $(MLP_SRC) $(LSTM_SRC) ../example/$@.c -o $(BIN)/$@ $(LIBS)



$(shell mkdir -p $(DIRS))

