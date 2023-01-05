#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "..\\LinearAlg\\linearAlg.h"
#include "..\\FileIO\\dataReader.h"

// does not include the input layer
#define LAYERS 3 
#define INPUT_LAYER_SIZE 4
#define HIDDEN_LAYER_SIZES 5, 5
#define OUTPUT_LAYER_SIZE 3

typedef struct neural_network
{
    float **weights[LAYERS];
    float **weightGradients[LAYERS]; // holds the changes to weights per iteration

    float *biases[LAYERS];
    float *biasGradients[LAYERS]; // holds the changes to biases per iteration

    float *activations[LAYERS];

    float** weight_product; //used in calculating the gradients
    float* weight_product_buffer; //used in calculating the weight product
} NeuralNet;

typedef struct training_set
{
    int training_examples;
    float **training_inputs;
    float **training_outputs;
} TrainingSet;

typedef struct testing_set
{
    int testing_examples;
    float **testing_inputs;
    float **testing_outputs;
} TestingSet;

void nnet_print(NeuralNet* nnet);
NeuralNet* nnet_init();
TestingSet* nnet_testing_set_init(int num_testing_examples);
TrainingSet* nnet_training_set_init(int num_training_examples);

void nnet_free(NeuralNet *nnet);
void nnet_free_test_set(TestingSet *set);
void nnet_free_training_set(TrainingSet *set);

void nnet_reset_network(NeuralNet* nnet);

void nnet_load_data(TrainingSet* training_set, TestingSet* testing_set, char* fileName, char* delimiter, int bufferSize);

float* nnet_run_data(float inputs[INPUT_LAYER_SIZE], NeuralNet* nnet);

int nnet_backprop(NeuralNet* nnet, TrainingSet* training_set, int num_mini_batches, int iterations, float learn_rate);
int nnet_iterate_gradients(NeuralNet* nnet, float* training_input, float* training_output);
void nnet_subtract_gradients(NeuralNet* nnet, float learn_rate, int num_training_examples);

float nnet_total_cost(NeuralNet* nnet, TrainingSet* set);

float nnet_test_results(NeuralNet* nnet, TestingSet* test_set, int print_each_test, int print_results);

void nnet_print(NeuralNet* nnet);
int nnet_save_to_file(NeuralNet* nnet, const char* fileName);
int nnet_load_from_file(NeuralNet* nnet, const char* fileName);

#endif