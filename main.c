#include "stdio.h"
#include "neural_net.h"
#include <profileapi.h>
//#include <omp.h>
#include <time.h>

#define LEARN_RATE 0.003
#define ITERATIONS 1000

#define NUM_TRAINING_EXAMPLES 120     // number of data points used for training
#define NUM_TESTING_EXAMPLES 30       // number of data points used for testing

#define MINI_BATCHES 8 //this should be a multiple of your maximum threads

int main()
{
    srand(time(NULL));
    NeuralNet* nnet = nnet_init();

    TrainingSet* training_set = nnet_training_set_init(NUM_TRAINING_EXAMPLES);
    TestingSet* test_set = nnet_testing_set_init(NUM_TESTING_EXAMPLES);
    nnet_load_data(training_set, test_set, "Data/iris.txt", ",", 63);
    
    LARGE_INTEGER tps;
    LARGE_INTEGER t1, t2;
    float timeDiff;
    QueryPerformanceFrequency(&tps);
    QueryPerformanceCounter(&t1);

    nnet_backprop_parallel(nnet, training_set, MINI_BATCHES, ITERATIONS, LEARN_RATE);
    
    QueryPerformanceCounter(&t2);
    timeDiff = (float)(t2.QuadPart - t1.QuadPart) / tps.QuadPart;
    printf("\ntraining time: %.5fs\n", timeDiff);

    nnet_test_results(nnet, test_set, 0, 1);

    // nnet_save_to_file(nnet, "bin/testNet.nnet");

    // nnet_load_from_file(nnet, "bin/testNet.nnet");
    // nnet_test_results(nnet, test_set, 0, 1);

    printf("\nfin\n");
}