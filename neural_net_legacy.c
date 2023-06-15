#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "linear_alg.h"
#include "data_handler.h"

// LAYERS does not include the input layer
#define LAYERS 3

// handwriten numbers:
#define INPUT_LAYER_SIZE 4
#define HIDDEN_LAYER_SIZES 5, 5
#define OUTPUT_LAYER_SIZE 3

// iris:
// #define INPUT_LAYER_SIZE 4
// #define HIDDEN_LAYER_SIZES 5, 5
// #define OUTPUT_LAYER_SIZE 3

#define MAX_THREADS 8

/*  the derivative of the activation
 *  relu : (a > 0)
 *  sigmoid : (sigmoid(a) * (1 - sigmoid(a)))
 *  none: (1)
 */
#define ACTIVATION_FUNCTION_DERIV(a) (sigmoid(a) * (1 - sigmoid(a)))

/*  activation function
 *  relu : (a*(a>0))
 *  sigmoid : (sigmoid(a))
 *  none: (a)
 */
#define ACTIVATION_FUNCTION(a) (sigmoid(a))

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

const int npl[LAYERS + 1] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE};

void nnet_layer_function_dense(float **weights, int rows, int columns, float *preLayer, float *bias, float *destination)
{
    int i;
    float pre_activation = 0;
    for (i = 0; i < rows; i++)
    {
        pre_activation = laa_dot(preLayer, weights[i], columns) + bias[i];
        destination[i] = ACTIVATION_FUNCTION(pre_activation);
    }
}

void multiply_replace(float **a_matrixVals, int a_rows, int a_columns, float **b_matrixVals, int b_rows, int b_columns, float *buffer)
{
    int i, j, k;
    float sum = 0;
    if (a_columns != b_rows)
    {
        printf("matrices must have compatable dimensions!\n");
        exit(0);
    }

    for (i = 0; i < a_rows; i++)
    {
        sum = 0;
        for (k = 0; k < a_columns; k++)
        {
            buffer[k] = a_matrixVals[i][k];
            sum += a_matrixVals[i][k] * b_matrixVals[k][0];
        }
        a_matrixVals[i][0] = sum;
        for (j = 1; j < b_columns; j++)
        {
            sum = 0;
            for (k = 0; k < a_columns; k++)
            {
                sum += buffer[k] * b_matrixVals[k][j];
            }
            a_matrixVals[i][j] = sum;
        }
    }
}

float *nnet_feed_forward(float inputs[npl[0]], NeuralNet *nnet, float *activations[LAYERS])
{
    nnet_layer_function_dense(nnet->weights[0], npl[1], npl[0], inputs, nnet->biases[0], activations[0]);
    int i;
    // #pragma omp parallel for
    for (i = 1; i < LAYERS; i++)
    {
        nnet_layer_function_dense(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);
    }
    //laa_printVector(activations[LAYERS-1], npl[LAYERS]);
    return activations[LAYERS - 1];
}

void nnet_subtract_gradients(NeuralNet *nnet, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float learn_rate, int num_training_examples)
{
    //(RELU(activations[layer][i]))
    int layer, i, j;
    for (layer = 0; layer < LAYERS; layer++)
    {
        // laa_printVector(bias_gradients[layer], npl[layer+1]);
        // laa_printMatrix(weight_gradients[layer], npl[layer+1], npl[layer]);
        for (i = 0; i < npl[layer + 1]; i++)
        {
            for (j = 0; j < npl[layer]; j++)
            {
                //printf("%f\n", weight_gradients[layer][i][j]);
                nnet->weights[layer][i][j] -= weight_gradients[layer][i][j] * learn_rate / num_training_examples;
                weight_gradients[layer][i][j] = 0;
            }
            nnet->biases[layer][i] -= bias_gradients[layer][i] * learn_rate / num_training_examples;
            bias_gradients[layer][i] = 0;
        }
    }
}

void update_last_two_layer_gradients(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float *expected_output)
{
    int i, j, k;
    float diff, sum, dotSum;
    for (i = 0; i < npl[LAYERS]; i++)
    {
        diff = activations[LAYERS - 1][i] - expected_output[i];
        bias_gradients[LAYERS - 1][i] += diff * ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 1][i]);
        // printf("b%d[%d]: %f\n", LAYERS - 1, i, bias_gradients[LAYERS - 1][i]);
        for (j = 0; j < npl[LAYERS - 1]; j++)
        {
            weight_gradients[LAYERS - 1][i][j] += diff * activations[LAYERS - 2][j] * ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 1][i]);

            bias_gradients[LAYERS - 2][j] += nnet->weights[LAYERS - 1][i][j] * diff * ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 2][i]);

            dotSum = 0;
            for (k = 0; k < npl[LAYERS - 2]; k++)
            {
                dotSum += nnet->weights[LAYERS - 1][i][j] * diff * activations[LAYERS - 3][k];
            }
            weight_gradients[LAYERS - 2][j][k] += dotSum * ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 2][i]);
        }
    }
}

int nnet_adjust_gradients(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float **weight_product, float *weight_product_buffer, float *training_input, float *training_output)
{
    int layer;
    int i, j, k, l;
    float dotSum1 = 0;
    float dotSum2 = 0;

    nnet_feed_forward(training_input, nnet, activations);

    // compute the last layer two gradients so that the rest can be calculated in a simple algorithm
    update_last_two_layer_gradients(nnet, activations, weight_gradients, bias_gradients, training_output);

    // set the weight product to the last layer's weight
    laa_copyMatrixValues(nnet->weights[LAYERS - 1], weight_product, npl[LAYERS], npl[LAYERS - 1]);

    for (layer = LAYERS - 2; layer > 1; layer--)
    {
        multiply_replace(weight_product, npl[layer + 2], npl[layer + 1], nnet->weights[layer], npl[layer + 1], npl[layer], weight_product_buffer);
        for (i = 0; i < npl[layer]; i++)
        {
            for (j = 0; j < npl[layer - 1]; j++)
            {
                dotSum1 = 0;
                for (k = 0; k < npl[LAYERS]; k++)
                {
                    dotSum1 += weight_product[k][i] * activations[layer - 1][j] * (activations[LAYERS - 1][k] - training_output[k]);
                }
                weight_gradients[layer - 1][i][j] += dotSum1 * ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
            }

            dotSum2 = 0;
            for (k = 0; k < npl[LAYERS]; k++)
            {
                dotSum2 += weight_product[k][i] * (activations[LAYERS - 1][k] - training_output[k]);
            }
            bias_gradients[layer - 1][i] += dotSum2 * ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
        }
    }
    multiply_replace(weight_product, npl[layer + 2], npl[layer + 1], nnet->weights[layer], npl[layer + 1], npl[layer], weight_product_buffer);
    for (i = 0; i < npl[layer]; i++)
    {
        for (j = 0; j < npl[layer - 1]; j++)
        {
            dotSum1 = 0;
            for (k = 0; k < npl[LAYERS]; k++)
            {
                dotSum1 += weight_product[k][i] * training_input[j] * (activations[LAYERS - 1][k] - training_output[k]);
            }
            weight_gradients[layer - 1][i][j] += dotSum1 * ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
        }

        dotSum2 = 0;
        for (k = 0; k < npl[LAYERS]; k++)
        {
            dotSum2 += weight_product[k][i] * (activations[LAYERS - 1][k] - training_output[k]);
        }
        bias_gradients[layer - 1][i] += dotSum2 * ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
    }
}

int nnet_optimize2(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int iterations, float learn_rate)
{
    printf("initializing backprop... ");
    const int examples_per_batch = num_mini_batches ? training_set->num_examples / num_mini_batches : 0;
    int iteration;
    int nthExample;
    int layer;
    int batch;
    int i;
    int largest_layer_size = 0;
    float **weight_gradients[LAYERS];
    float *bias_gradients[LAYERS];
    float *activations[LAYERS];
    float **weight_product;
    float *weight_product_buffer;

    for (i = 0; i < LAYERS; i++)
    {
        if (npl[i] > largest_layer_size)
            largest_layer_size = npl[i];
        weight_gradients[i] = laa_allocMatrix(npl[i + 1], npl[i], 0);
        bias_gradients[i] = laa_allocVector(npl[i + 1], 0);
        activations[i] = laa_allocVector(npl[i + 1], 0);
    }

    if (npl[i] > largest_layer_size)
        largest_layer_size = npl[i];
    weight_product = laa_allocMatrixRaw(largest_layer_size, largest_layer_size);
    weight_product_buffer = laa_allocVectorRaw(largest_layer_size);
    printf("done\n");

    for (iteration = iterations; iteration--;)
    {
        printf("\rtraining... %d/%d cost: %f", iterations-iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < num_mini_batches; batch++)
        {
            for (nthExample = batch * examples_per_batch; nthExample < (batch + 1) * examples_per_batch; nthExample++)
            {
                nnet_adjust_gradients(nnet, activations, weight_gradients, bias_gradients, weight_product, weight_product_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
            }
            nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, examples_per_batch);
        }
        for (; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_adjust_gradients(nnet, activations, weight_gradients, bias_gradients, weight_product, weight_product_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, examples_per_batch);

        if (isnan(activations[0][0]))
        {
            printf("oh no!");
            return 0;
        }
    }
    printf("done\n");

    for (i = 0; i < LAYERS; i++)
    {
        laa_freeMatrix(weight_gradients[i], npl[i + 1]);
        laa_freeVector(bias_gradients[i]);
        laa_freeVector(activations[i]);
    }
    laa_freeMatrix(weight_product, largest_layer_size);
    laa_freeVector(weight_product_buffer);
    return 1;
}

int nnet_optimize_parallel2(NeuralNet *nnet, TrainingSet *training_set, int parallel_batches, int iterations, float learn_rate)
{
    printf("initializing backprop... ");
    const int examples_per_thread = training_set->num_examples / MAX_THREADS / parallel_batches;
    int iteration;
    int layer;
    int thread;
    int batch;
    int i, j;
    int largest_layer_size = 0;
    float **weight_gradients[MAX_THREADS][LAYERS];
    float *bias_gradients[MAX_THREADS][LAYERS];
    float *activations[MAX_THREADS][LAYERS];
    float **weight_product[MAX_THREADS];
    float *weight_product_buffer[MAX_THREADS];
    for (i = 0; i <= LAYERS; i++)
        if (npl[i] > largest_layer_size)
            largest_layer_size = npl[i];

    for (i = 0; i < MAX_THREADS; i++)
    {
        for (j = 0; j < LAYERS; j++)
        {
            weight_gradients[i][j] = laa_allocMatrix(npl[j + 1], npl[j], 0);
            bias_gradients[i][j] = laa_allocVector(npl[j + 1], 0);
            activations[i][j] = laa_allocVector(npl[j + 1], 0);
        }
        weight_product[i] = laa_allocMatrixRaw(largest_layer_size, largest_layer_size);
        weight_product_buffer[i] = laa_allocVectorRaw(largest_layer_size);
    }
    printf("done\n");

    for (iteration = iterations; iteration--;)
    {
        // printf("\r%d/%d", iterations-iteration, iterations);
        //printf("\rtraining... %d/%d cost: %f", iterations - iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < parallel_batches; batch++)
        {
#pragma omp parallel for
            for (thread = 0; thread < MAX_THREADS; thread++)
            {
                for (int nthExample = (batch * MAX_THREADS + thread) * examples_per_thread; nthExample < (batch * MAX_THREADS + thread + 1) * examples_per_thread; nthExample++)
                {
                    nnet_adjust_gradients(nnet, activations[thread], weight_gradients[thread], bias_gradients[thread], weight_product[thread], weight_product_buffer[thread], training_set->inputs[nthExample], training_set->outputs[nthExample]);
                }
            }

            for (i = 0; i < MAX_THREADS; i++)
            {
                nnet_subtract_gradients(nnet, weight_gradients[i], bias_gradients[i], learn_rate, training_set->num_examples);
            }
        }

        for (int nthExample = parallel_batches * MAX_THREADS * examples_per_thread; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_adjust_gradients(nnet, activations[0], weight_gradients[0], bias_gradients[0], weight_product[0], weight_product_buffer[0], training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients[0], bias_gradients[0], learn_rate, training_set->num_examples);

        if (isnan(activations[LAYERS - 1][0][0]))
        {
            printf("oh no!");
            return 0;
        }
    }

    for (i = 0; i < MAX_THREADS; i++)
    {
        for (j = 0; j < LAYERS; j++)
        {
            laa_freeMatrix(weight_gradients[i][j], npl[j + 1]);
            laa_freeVector(bias_gradients[i][j]);
            laa_freeVector(activations[i][j]);
        }
        laa_freeMatrix(weight_product[i], largest_layer_size);
        laa_freeVector(weight_product_buffer[i]);
    }
    printf("done\n");
    return 1;
}
