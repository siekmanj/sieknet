CC=gcc

CPULIBOUT=libsieknetcpu.so
GPULIBOUT=libsieknetgpu.so

DIRS=bin model data log

BIN=bin

SRC_DIR=src
DAT_DIR=data
MJ_DIR=$(HOME)/.mujoco/mujoco200_linux

INCLUDE=-Iinclude -Ienv
LIBS=-lm 
GPULIBS=$(LIBS) -lOpenCL
MJLIBS=-lmujoco200 -lGL -lglew $(MJ_DIR)/bin/libglfw.so.3

CFLAGS=-O3 -Wno-unused-function
GPUFLAGS=$(CFLAGS) -DSIEKNET_USE_GPU
MUJOCOFLAGS=$(CFLAGS) -I$(MJ_DIR)/include -L$(MJ_DIR)/bin

LSTM_SRC=$(SRC_DIR)/lstm.c
RNN_SRC=$(SRC_DIR)/rnn.c
MLP_SRC=$(SRC_DIR)/mlp.c
OPTIM_SRC=$(SRC_DIR)/optimizer.c
ENV_SRC=$(SRC_DIR)/env.c
CL_SRC=$(SRC_DIR)/opencl_utils.c

MNIST_SRC=$(SRC_DIR)/mnist.c
GA_SRC=$(SRC_DIR)/ga.c
ARS_SRC=$(SRC_DIR)/ars.c
HOPPER_SRC=env/hopper_env.c

CPU_SRC=$(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(OPTIM_SRC) $(GA_SRC) $(ARS_SRC) $(ENV_SRC)
GPU_SRC=$(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(OPTIM_SRC) $(GA_SRC) $(ARS_SRC) $(ENV_SRC) $(CL_SRC)

bug:
	$(CC) cassietest.c -I./env/cassie/include -L./bin -lcassiemujoco 

cpu: src/*.c
	gcc -shared -o $(BIN)/$(CPULIBOUT) -fPIC $(CFLAGS) $(CPU_SRC) $(INCLUDE) $(LIBS) -Wl,-rpath bin/ -fopenmp

gpu: src/*.c
	gcc -shared -o $(BIN)/$(GPULIBOUT) -fPIC $(GPUFLAGS) $(GPU_SRC) $(INCLUDE) $(GPULIBS) -Wl,-rpath bin/

clean:
	rm ./bin/*



# DEPRECATED

char:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(RNN_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
char_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(RNN_SRC) $(MLP_SRC) $(CL_SRC) example/char.c -o $(BIN)/$@ $(GPULIBS)

mlp_mnist:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
mlp_mnist_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/mlp_mnist.c -o $(BIN)/$@ $(GPULIBS)

binary:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
binary_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(CL_SRC) example/binary.c -o $(BIN)/$@ $(GPULIBS)

clock:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
clock_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(CL_SRC) example/clock.c -o $(BIN)/$@ $(GPULIBS)

sequence:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
sequence_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(CL_SRC) example/sequence.c -o $(BIN)/$@ $(GPULIBS)

genetic:
	$(CC) $(MUJOCOFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(GA_SRC) $(HOPPER_SRC) example/$@.c $(MJLIBS) -o $(BIN)/$@ $(LIBS)

search:
	$(CC) $(MUJOCOFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(RNN_SRC) $(LSTM_SRC) $(RS_SRC) env/*.c example/$@.c $(MJLIBS) -o $(BIN)/$@ $(LIBS) -fopenmp -DNUM_THREADS=4


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

