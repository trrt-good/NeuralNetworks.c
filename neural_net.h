/*
This header file contains declarations for a neural network implemented in C.
- The NeuralNet struct contains the weights and biases for each layer, as well as the activations for each layer.
- It also contains arrays to hold the gradients for the weights and biases, as well as temporary variables used in calculating the gradients.
- The TrainingSet and TestingSet structs contain the data for training and testing the network, including the number of examples and the inputs/outputs for each example.
- Functions are provided for initializing and freeing the NeuralNet, TrainingSet, and TestingSet, as well as resetting the weights and biases in the NeuralNet.
- There are also functions for loading data from a file and running the network on a given set of inputs.
- The main training function is nnet_backprop, which performs backpropagation using mini-batches and a specified number of iterations.
- There are also functions for iterating through the gradients for a single training example, subtracting the gradients from the weights and biases, and calculating the total cost for a set of training examples.
- The nnet_test_results function can be used to test the accuracy of the network on a given TestingSet, with the option to print the results for each test.
- Finally, there are functions for printing and saving/loading the network to/from a file.
*/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "LinearAlg\\linearAlg.h"
#include "DataReader\\dataReader.h"

// LAYERS does not include the input layer
#define LAYERS 3
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZES 40, 20
#define OUTPUT_LAYER_SIZE 10
#define MAX_THREADS 8

/*
The neural_network struct represents a neural network with multiple layers.
It has weights and biases for each layer, as well as gradients for each weight and bias.
It also has activations for each layer, as well as intermediate values used in calculating gradients.
*/
typedef struct neural_network
{
    float **weights[LAYERS];
    float *biases[LAYERS];
} NeuralNet;

/*
The training_set struct represents a set of data used for training a neural network.
It has the number of examples in the set, as well as the inputs and corresponding outputs for each example.
*/
typedef struct training_set
{
    int num_examples;
    float **inputs;
    float **outputs;
} TrainingSet;

/*
The testing_set struct represents a set of data used for testing a trained neural network.
It has the number of examples in the set, as well as the inputs and corresponding outputs for each example.
*/
typedef struct testing_set
{
    int num_examples;
    float **inputs;
    float **outputs;
} TestingSet;

void nnet_print(NeuralNet* nnet);
NeuralNet* nnet_init(float scale_factor);
TestingSet* nnet_testing_set_init(int num_testing_examples);
TrainingSet* nnet_training_set_init(int num_training_examples);
void nnet_free(NeuralNet *nnet);
void nnet_free_test_set(TestingSet *set);
void nnet_free_training_set(TrainingSet *set);
void nnet_reset_network(NeuralNet* nnet);
void nnet_load_data(TrainingSet *training_set, TestingSet *testing_set);
void nnet_shuffle_data(float **inputs, float **outputs, int n);
float* nnet_run_data(float inputs[INPUT_LAYER_SIZE], NeuralNet* nnet, float* activations[LAYERS]);
float nnet_total_cost(NeuralNet* nnet, float** inputs, float** outputs, int n);
int nnet_backprop(NeuralNet* nnet, TrainingSet* training_set, int num_mini_batches, int iterations, float learn_rate);
int nnet_backprop_parallel(NeuralNet *nnet, TrainingSet *training_set, int parallel_batches, int iterations, float learn_rate);
int nnet_iterate_gradients(NeuralNet *nnet, float* activations[LAYERS], float** weight_gradients[LAYERS], float* bias_gradients[LAYERS], float** weight_product, float* weight_product_buffer, float *training_input, float *training_output);
void nnet_subtract_gradients(NeuralNet *nnet, float** weight_gradients[LAYERS], float* bias_gradients[LAYERS], float learn_rate, int num_training_examples);
float nnet_test_results(NeuralNet* nnet, TestingSet* test_set, int print_each_test, int print_results);
void nnet_print(NeuralNet* nnet);
int nnet_save_to_file(NeuralNet* nnet, const char* fileName);
int nnet_load_from_file(NeuralNet* nnet, const char* fileName);

#endif