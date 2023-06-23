#include "stdio.h"
#include "neural_net.h"
#include <profileapi.h>
//#include <omp.h>
#include <time.h>

// best mnist configuration (that I have tested):
#define LEARN_RATE 0.05
#define EPOCHS 10
 
#define NUM_TRAINING_EXAMPLES 60000     // number of data points used for training
#define NUM_TESTING_EXAMPLES 10000       // number of data points used for testing
 
#define BATCHES 150

#define INIT_MIN -0.1
#define INIT_MAX 0.1

// #define LEARN_RATE 0.01
// #define EPOCHS 500
 
// #define NUM_TRAINING_EXAMPLES 120     // number of data points used for training
// #define NUM_TESTING_EXAMPLES 30       // number of data points used for testing
 
// #define BATCHES 10

// #define INIT_MIN -0.5
// #define INIT_MAX 0.5

int main()
{
    srand(time(NULL));
    NeuralNet* nnet = nnet_init(INIT_MIN, INIT_MAX);

    printf("loading data... ");

    TrainingSet* training_set = dh_training_set_init(NUM_TRAINING_EXAMPLES);
    TestingSet* test_set = dh_testing_set_init(NUM_TESTING_EXAMPLES);

    //dh_read_data_iris("Data/iris.data", training_set, test_set);

    dh_read_mnist_digits_images("Data/train-images.idx3-ubyte", training_set->num_examples, training_set->inputs);
    dh_read_mnist_digits_labels("Data/train-labels.idx1-ubyte", training_set->num_examples, training_set->outputs);

    dh_read_mnist_digits_images("Data/t10k-images.idx3-ubyte", test_set->num_examples, test_set->inputs);
    dh_read_mnist_digits_labels("Data/t10k-labels.idx1-ubyte", test_set->num_examples, test_set->outputs);

    printf("done\n");

    LARGE_INTEGER tps;
    LARGE_INTEGER t1, t2;
    float timeDiff;
    QueryPerformanceFrequency(&tps);
    QueryPerformanceCounter(&t1);

    nnet_optimize_parallel(nnet, training_set, BATCHES, EPOCHS, LEARN_RATE);
    
    QueryPerformanceCounter(&t2);
    timeDiff = (float)(t2.QuadPart - t1.QuadPart) / tps.QuadPart;
    printf("\ntraining time: %.5fs\n", timeDiff);

    printf("final cost: %f\n", nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
    (void)nnet_test_results(nnet, test_set, 0, 1);
        
    nnet_save_to_file(nnet, "bin/testNet.nnet");

    // nnet_load_from_file(nnet, "bin/testNet.nnet");
    // nnet_test_results(nnet, test_set, 0, 1);

    printf("\nfin\n");
}