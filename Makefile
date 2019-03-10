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

CFLAGS=-O3
GPUFLAGS=$(CFLAGS) -DGPU

LSTM_SRC=$(SRC_DIR)/lstm.c
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

#char:
#	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
#shakespeare:
#	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(LSTM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
mlp_mnist:
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
mlp_mnist_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MNIST_SRC) $(MLP_SRC) $(CL_SRC) example/mlp_mnist.c -o $(BIN)/mlp_mnist $(GPULIBS)
#binary:
#	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
#sequence:
#	$(CC) $(CFLAGS) $(INCLUDE) $(OPTIM_SRC) $(MLP_SRC) $(LSTM_SRC) example/$@.c -o $(BIN)/$@ $(LIBS)
#test_cpu:
#	$(CC) -o bin/test -Iinclude example/test.c -L./bin -lcpusieknet -lm
test_gpu:
	$(CC) $(GPUFLAGS) $(INCLUDE) $(SRC_DIR)/*.c example/test.c -o $(BIN)/test $(GPULIBS)
#test_cpu:
#	$(CC) -o bin/test $(INCLUDE) $(SRC_DIR)/*.c example/test.c $(LIBS)
#test_gpu:
#	$(CC) -o bin/test $(INCLUDE) $(SRC_DIR)/*.c example/test.c $(GPULIBS) -DGPU

$(shell mkdir -p $(DIRS))

