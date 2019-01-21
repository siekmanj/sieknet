/* Jonah Siekmann
 * 7/16/2018
 * This is an interface for the MNIST handwritten digits image dataset, for use in a neural network.
 */


#include "./mnist.h"

/*
 * 
 */
uint32_t extractInt32(uint8_t *ptr, size_t offset){
	uint32_t num = 0;
	for(int i = offset; i < offset+4; i++){
		num |= ptr[i] << (3-i)*8;
	}
	return num;
}

void printImage(ImageSet *imgset, size_t index){
	size_t rows = imgset->height;
	size_t cols = imgset->width;
	size_t size = rows*cols;
	int base = index*size + 16;
	for(int i = base; i < base+rows*cols; i++){
		uint8_t val = imgset->imgBuff[i];
		if(((i-base) % cols) == 0) printf("\n");
		if(val < 5) printf(".");
		else if(val < 100) printf(":");
		else if(val < 175) printf("l");
		else if(val < 255) printf("X");
	}
	printf("\n");
}

int openImageSet(ImageSet *imgset, size_t size, char* imgFilename, char* labelFilename){
	FILE *imageFile = fopen(imgFilename, "rb");
	FILE *labelFile = fopen(labelFilename, "rb");
	if(!imageFile){
		printf("Could not open '%s'\n", imgFilename);
		exit(1);
	}
	else if(!labelFile){
		printf("Could not open '%s'\n", labelFilename);
		exit(1);
	}

	uint8_t *imgBuff = malloc(sizeof(uint8_t)*size);
	uint8_t *labelBuff = malloc(sizeof(uint8_t)*size);

	size_t i = fread(imgBuff, sizeof(uint8_t), size, imageFile);
	size_t l = fread(labelBuff, sizeof(uint8_t), size, labelFile);
	if(i != size) return 0;

	imgset->magic_num = extractInt32(imgBuff, 0);
	imgset->numImages = extractInt32(imgBuff, 4);
	imgset->height = extractInt32(imgBuff, 8);
	imgset->width = extractInt32(imgBuff, 12);
	imgset->imgBuff = imgBuff;
	imgset->labelBuff = labelBuff;
	return 1;
}

float* img2floatArray(ImageSet *imgset, int index, size_t *rows, size_t *cols){
	*rows = imgset->height;
	*cols = imgset->width;
	int base = index * imgset->height * imgset->width + 16;
	float *array = (float*)malloc(imgset->height*imgset->width*sizeof(float));
	for(int k = 0; k < imgset->height*imgset->width; k++){
		array[k] = (float)imgset->imgBuff[base + k] / 255;
	}
	return array;
}
int label(ImageSet *imgset, int index){
	return imgset->labelBuff[8 + index];
}
