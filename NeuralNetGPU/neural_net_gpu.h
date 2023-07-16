/*
This header file contains declarations for a neural network implemented in C.
- The NeuralNet struct contains the weights and biases for each layer, as well as the activations for each layer.
- It also contains arrays to hold the gradients for the weights and biases, as well as temporary variables used in calculating the gradients.
- The TrainingSet and TestingSet structs contain the data for training and testing the network, including the number of examples and the inputs/outputs for each example.
- Functions are provided for initializing and freeing the NeuralNet, TrainingSet, and TestingSet, as well as resetting the weights and biases in the NeuralNet.
- There are also functions for loading data from a file and running the network on a given set of inputs.
- The main training function is nnet_optimize, which performs backpropagation using mini-batches and a specified number of iterations.
- There are also functions for iterating through the gradients for a single training example, subtracting the gradients from the weights and biases, and calculating the total cost for a set of training examples.
- The nnet_test_results function can be used to test the accuracy of the network on a given TestingSet, with the option to print the results for each test.
- Finally, there are functions for printing and saving/loading the network to/from a file.
*/

#ifndef NEURAL_NETWORK_GPU_H
#define NEURAL_NETWORK_GPU_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "linear_alg.h"
#include "data_handler.h"

#define MAX_LAYER_SIZE 32
#define BLOCK_SIZE 32

// LAYERS value does not count the input layer
#define LAYERS 3

// mnist configuration: 
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZES 30, 20
#define OUTPUT_LAYER_SIZE 10

// #define INPUT_LAYER_SIZE 4
// #define HIDDEN_LAYER_SIZES 5, 5
// #define OUTPUT_LAYER_SIZE 3

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

void nnet_print(NeuralNet* nnet);
NeuralNet* nnet_init(float init_min, float init_max);

void nnet_free(NeuralNet *nnet);
void nnet_reset_network(NeuralNet* nnet);
void nnet_feed_forward(float *d_inputs, float *d_weights[LAYERS], float *d_biases[LAYERS], float *d_activations[LAYERS]);
float nnet_total_cost(float **correct_outputs, float ** predictions, int num_data_points);
int nnet_optimize(NeuralNet* nnet, TrainingSet* training_set, int num_mini_batches, int iterations, float learn_rate);
void nnet_free_gpu_wba(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float *d_activations[LAYERS]);
void nnet_alloc_gpu_wba(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float *d_activations[LAYERS]);
void nnet_alloc_gpu_data(float *d_training_inputs, float *d_training_outputs, int num_examples);
void nnet_free_gpu_data(float *d_training_inputs, float *d_training_outputs);
float nnet_test_results(NeuralNet* nnet, TestingSet* test_set, int print_each_test, int print_results);
void nnet_print(NeuralNet* nnet);
int nnet_save_to_file(NeuralNet* nnet, const char* fileName);
int nnet_load_from_file(NeuralNet* nnet, const char* fileName);

#endif