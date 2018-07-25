/* Jonah Siekmann
 * 7/16/2018
 * This is an interface for the MNIST handwritten digits image dataset, for use in a neural network.
 */
#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef struct mnist{
	uint32_t magic_num;
	uint32_t numImages;
	uint32_t height;
	uint32_t width;
	uint8_t *imgBuff;
	uint8_t *labelBuff;
} ImageSet;

uint32_t extractInt32(uint8_t*, size_t);
void printImage(ImageSet*, size_t);
int openImageSet(ImageSet*, size_t, char*, char*);
float* img2floatArray(ImageSet*, int, size_t*, size_t*);
int label(ImageSet*, int);

#endif
