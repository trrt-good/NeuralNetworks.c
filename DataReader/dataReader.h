#ifndef DATA_READER_H
#define DATA_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int read_mnist_number_data(char *filename, int dataPoints, float **inputs, float **outputs);
int read_iris_data(char* fileName, char* delimiter, int bufferSize, int totalDataPoints, int numTesting, int numInputs, int numOutputs, float **trainingInputArr, float **trainingOutputsArr, float **testingInputArr, float **testingOutputArr, int randomizeOrder);
#endif