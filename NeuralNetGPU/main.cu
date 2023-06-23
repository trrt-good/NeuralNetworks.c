#include "stdio.h"
#include "neural_net_gpu.h"
#include <windows.h>
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
    TestingSet* testing_set = dh_testing_set_init(NUM_TESTING_EXAMPLES);

    //dh_read_data_iris("Data/iris.data", training_set, testing_set);

    dh_read_mnist_digits_images("Data/train-images.idx3-ubyte", training_set->num_examples, training_set->inputs);
    dh_read_mnist_digits_labels("Data/train-labels.idx1-ubyte", training_set->num_examples, training_set->outputs);

    dh_read_mnist_digits_images("Data/t10k-images.idx3-ubyte", testing_set->num_examples, testing_set->inputs);
    dh_read_mnist_digits_labels("Data/t10k-labels.idx1-ubyte", testing_set->num_examples, testing_set->outputs);

    printf("done\n");

    nnet_load_data_to_GPU(training_set);
    nnet_load_nnet_to_GPU(nnet);

    // start timing
    LARGE_INTEGER frequency;
    LARGE_INTEGER start, end;
    double interval;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    //nnet_optimize(nnet, training_set, BATCHES, EPOCHS, LEARN_RATE);
    
    // end timing
    QueryPerformanceCounter(&end);
    interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("\ntraining time: %.6fs\n", interval);

    nnet_load_nnet_from_GPU(nnet);

    printf("final cost: %f\n", nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
    (void)nnet_test_results(nnet, testing_set, 0, 1);
        
    nnet_save_to_file(nnet, "bin/testNet.nnet");

    //free memory
    nnet_free_gpu_nnet_and_activations();
    nnet_free_gpu_data();
    nnet_free(nnet);
    dh_free_testing_set(testing_set);
    dh_free_training_set(training_set);

    // nnet_load_from_file(nnet, "bin/testNet.nnet");
    // nnet_testing_results(nnet, testing_set, 0, 1);

    printf("\nfin\n");
}