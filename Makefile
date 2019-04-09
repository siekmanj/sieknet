CC=gcc

CPULIBOUT=libcpusieknet.so
GPULIBOUT=libgpusieknet.so

DIRS=bin model data log

BIN=bin

SRC_DIR=src
DAT_DIR=data

INCLUDE=-Iinclude
LIBS=-lm 
GPULIBS=$(LIBS) -lOpenCL

CFLAGS=-O0 -Wall -Wno-unused-function
GPUFLAGS=$(CFLAGS) -DSIEKNET_USE_GPU

LSTM_SRC=$(SRC_DIR)/lstm.c
RNN_SRC=$(SRC_DIR)/rnn.c
MLP_SRC=$(SRC_DIR)/mlp.c
MNIST_SRC=$(SRC_DIR)/mnist.c
OPTIM_SRC=$(SRC_DIR)/optimizer.c
CL_SRC=$(SRC_DIR)/opencl_utils.c

CPU_SRC=$(MLP_SRC) $(OPTIM_SRC)
GPU_SRC=$(MLP_SRC) $(OPTIM_SRC) $(CL_SRC)

libcpu: src/*.c
	gcc -shared -o $(BIN)/$(CPULIBOUT) -fPIC $(CFLAGS) $(CPU_SRC) $(INCLUDE) $(LIBS) -Wl,-rpath /home/jonah/sieknet/bin

libgpu: src/*.c
	gcc -shared -o $(BIN)/$(GPULIBOUT) -fPIC $(GPUFLAGS) $(GPU_SRC) $(INCLUDE) $(GPULIBS) -Wl,-rpath /home/jonah/sieknet/bin

char:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
char_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) $(CL_SRC) example/char.c -o $(BIN)/$@ $(GPULIBS)

mlp_mnist:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
mlp_mnist_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/mlp_mnist.c -o $(BIN)/$@ $(GPULIBS)

binary:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
binary_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(CL_SRC) example/binary.c -o $(BIN)/$@ $(GPULIBS)

sequence:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
sequence_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(CL_SRC) example/sequence.c -o $(BIN)/$@ $(GPULIBS)

test_lstm:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) example/test_lstm.c -o $(BIN)/$@ $(LIBS)
test_lstm_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) $(CL_SRC) example/test_lstm.c -o $(BIN)/$@ $(GPULIBS)

test_rnn:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(RNN_SRC) $(MLP_SRC) example/test_rnn.c -o $(BIN)/$@ $(LIBS)
test_rnn_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(RNN_SRC) $(MLP_SRC) $(CL_SRC) example/test_rnn.c -o $(BIN)/$@ $(GPULIBS)

test_mlp:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) example/test_mlp.c -o $(BIN)/$@ $(LIBS)
test_mlp_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(CL_SRC) example/test_mlp.c -o $(BIN)/$@ $(GPULIBS)

$(shell mkdir -p $(DIRS))

