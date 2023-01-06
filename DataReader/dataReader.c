#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataReader.h"
#include "..\\LinearAlg\\linearAlg.h"

int getModCount(char *str)
{
    int i, count, len = strlen(str);
    for (i = 0; i < len; i ++)
    {
        if (str[i] == '%')
            count++;
    }
}

void swap(float** arr, int i, int j)
{
    if (i == j)
        return;
    float* temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

int read_iris_data(char* fileName, char* delimiter, int bufferSize, int totalDataPoints, int numTesting, int numInputs, int numOutputs, float **trainingInputArr, float **trainingOutputsArr, float **testingInputArr, float **testingOutputArr, int randomizeOrder)
{
    srand(time(NULL));
    float** allInputs = laa_allocMatrix(totalDataPoints, numInputs, 0);
    float** allOutputs = laa_allocMatrix(totalDataPoints, numOutputs, 0);
    char strBuffer[bufferSize];
    char* stringToken;
    int i, j, nthExample = 0;
    int randIndex;
        
    FILE *filePointer; 

    filePointer = fopen(fileName, "r");
    if (filePointer == NULL)
    {
        printf("could not find file! %s", fileName);
    }

    while(fgets(strBuffer, bufferSize+1, filePointer) != NULL && nthExample < totalDataPoints)
    {
        stringToken = strtok(strBuffer, delimiter);
        for (i = 0; i < numInputs && stringToken != NULL; i ++)
        {
            allInputs[nthExample][i] = (float)atof(stringToken);
            
            stringToken = strtok(NULL, delimiter);
        }
        for (i = 0; i < numOutputs && stringToken != NULL; i ++)
        {
            allOutputs[nthExample][i] = (float)atof(stringToken);
        
            stringToken = strtok(NULL, delimiter);
        }
        if (randomizeOrder)
        {
            randIndex = (nthExample)? nthExample-rand()%nthExample : 0;
            swap(allInputs, nthExample, randIndex);
            swap(allOutputs, nthExample, randIndex);
        }
        nthExample++;
    }
    for (i = 0; i < totalDataPoints-numTesting; i ++)
    {
        for (j = 0; j < numInputs; j++)
        {
            trainingInputArr[i][j] = allInputs[i][j];
        }
        for (j = 0; j < numOutputs; j++)
        {
            trainingOutputsArr[i][j] = allOutputs[i][j];
        }
    }
    if (numTesting)
    {
        for (i = totalDataPoints-numTesting; i < totalDataPoints; i ++)
        {
            for (j = 0; j < numInputs; j++)
            {
                testingInputArr[i-totalDataPoints+numTesting][j] = allInputs[i][j];
            }
            for (j = 0; j < numOutputs; j++)
            {
                testingOutputArr[i-totalDataPoints+numTesting][j] = allOutputs[i][j];
            }
        }
    }
    laa_freeMatrix(allInputs, totalDataPoints);
    laa_freeMatrix(allOutputs, totalDataPoints);
    fclose(filePointer);
}

int read_iris_data(char* fileName, char* delimiter, int bufferSize, int totalDataPoints, int numTesting, int numInputs, int numOutputs, float **trainingInputArr, float **trainingOutputsArr, float **testingInputArr, float **testingOutputArr, int randomizeOrder)
{
    srand(time(NULL));
    float** allInputs = laa_allocMatrix(totalDataPoints, numInputs, 0);
    float** allOutputs = laa_allocMatrix(totalDataPoints, numOutputs, 0);
    char strBuffer[bufferSize];
    char* stringToken;
    int i, j, nthExample = 0;
    int randIndex;
        
    FILE *filePointer; 

    filePointer = fopen(fileName, "r");
    if (filePointer == NULL)
    {
        printf("could not find file! %s", fileName);
    }

    while(fgets(strBuffer, bufferSize+1, filePointer) != NULL && nthExample < totalDataPoints)
    {
        stringToken = strtok(strBuffer, delimiter);
        for (i = 0; i < numInputs && stringToken != NULL; i ++)
        {
            allInputs[nthExample][i] = (float)atof(stringToken);
            
            stringToken = strtok(NULL, delimiter);
        }
        for (i = 0; i < numOutputs && stringToken != NULL; i ++)
        {
            allOutputs[nthExample][i] = (float)atof(stringToken);
        
            stringToken = strtok(NULL, delimiter);
        }
        if (randomizeOrder)
        {
            randIndex = (nthExample)? nthExample-rand()%nthExample : 0;
            swap(allInputs, nthExample, randIndex);
            swap(allOutputs, nthExample, randIndex);
        }
        nthExample++;
    }
    for (i = 0; i < totalDataPoints-numTesting; i ++)
    {
        for (j = 0; j < numInputs; j++)
        {
            trainingInputArr[i][j] = allInputs[i][j];
        }
        for (j = 0; j < numOutputs; j++)
        {
            trainingOutputsArr[i][j] = allOutputs[i][j];
        }
    }
    if (numTesting)
    {
        for (i = totalDataPoints-numTesting; i < totalDataPoints; i ++)
        {
            for (j = 0; j < numInputs; j++)
            {
                testingInputArr[i-totalDataPoints+numTesting][j] = allInputs[i][j];
            }
            for (j = 0; j < numOutputs; j++)
            {
                testingOutputArr[i-totalDataPoints+numTesting][j] = allOutputs[i][j];
            }
        }
    }
    laa_freeMatrix(allInputs, totalDataPoints);
    laa_freeMatrix(allOutputs, totalDataPoints);
    fclose(filePointer);
}
