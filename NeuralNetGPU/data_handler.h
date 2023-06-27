#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

TestingSet *dh_testing_set_init(int num_testing_examples);
TrainingSet *dh_training_set_init(int num_training_examples);
void dh_free_testing_set(TestingSet *set);
void dh_free_training_set(TrainingSet *set);

void dh_shuffle_data(float** data_inputs, float **data_outputs, int num_data_points);

int dh_read_data_iris(const char *filename, TrainingSet* training_set, TestingSet* testing_set);

int dh_read_mnist_digits_images(const char *filename, int num_data_points, float **data);
int dh_read_mnist_digits_labels(const char *filename, int num_data_points, float **data);

void dh_print_image(float* pixels, int image_width);

int read_mnist_number_data(const char *filename, int dataPoints, float **inputs, float **outputs);

#endif