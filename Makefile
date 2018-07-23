NETDIR = ./net
MNIST = ./mnist
CFLAGS = -lm
CSRC = ./neural.c ${NETDIR}/*.c ${MNIST}/*.c
default:
	gcc ${CSRC} ${CFLAGS}
clean:
	rm ./a.out
