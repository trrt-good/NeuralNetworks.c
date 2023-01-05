#ifndef DATA_READER
#define DATA_READER

int readRowData_ML(char* fileName, char* delimiter, int bufferSize, int totalDataPoints, int numTesting, int numInputs, int numOutputs, float **trainingInputArr, float **trainingOutputsArr, float **testingInputArr, float **testingOutputArr, int randomizeOrder);
#endif