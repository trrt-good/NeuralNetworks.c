#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <windows.h>
#include <profileapi.h>
#include "..\\LinearAlg\\linearAlg.h"
#include "..\\FileIO\\dataReader.h"
#include "neuralNet.h"

#define TRAIN_MULTIPLE 1
// --- --- --- --- --- --- --- --- --- main --- --- --- --- --- --- --- --- ---

int main()
{
    int i;
    const unsigned int NUM_INTERATIONS = 500;

    initNetwork();
    loadData("MachineLearning/Data/iris.txt", ",", 63);

    LARGE_INTEGER tps;
    LARGE_INTEGER t1, t2;
    float timeDiff;
    QueryPerformanceFrequency(&tps);
    QueryPerformanceCounter(&t1);

#if TRAIN_MULTIPLE

    float total = 0;
    int sucess = 1;
    int diverges = 0;

    for (int i = 1; i <= NUM_INTERATIONS; i ++)
    {
        printf("\rtest %d/%d", i, NUM_INTERATIONS);
        do
        {
            resetNetwork();
            sucess = backProp(0);
            if (!sucess)
                diverges++;
        } 
        while (!sucess);
        total += comparePredictions(0, 0);
    }
    printf("\naverage accuracy of %d trials: %f\n", NUM_INTERATIONS, total/NUM_INTERATIONS);
    printf("diverges: %d\n", diverges);

#else

    backProp(0);
    comparePredictions(0, 1);
    //loadNetworkFromFile("irisNet.nnet", 6);

#endif

    QueryPerformanceCounter(&t2);
    timeDiff = (float)(t2.QuadPart - t1.QuadPart) / tps.QuadPart;
#if TRAIN_MULTIPLE 
    printf("average training time: %.5fs\n", timeDiff/NUM_INTERATIONS);
#else
    printf("training time: %.5fs\n", timeDiff);
#endif

    
    

    //saveToFile("irisNet.nnet");
    

    printf("\nfin");
    return 0;
}