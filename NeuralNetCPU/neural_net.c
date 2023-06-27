#include "neural_net.h"

/**
 * @brief npl = NODES PER LAYER. this is the central array
 * which the neural network is built from.
 * Index 0 of the array is how many input neurons there are.
 * The last index is how many output neurons there are.
 * There can be as many numbers inbetween as desired and each
 * symbolizes the number of hidden neurons there are in the layer
 * corresponding to it's index.
 */
const int npl[LAYERS + 1] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE};

// function declairations
void multiply_MtVreplace(float **matrixVals, int rows, int cols, float *vectorVals, float *buffer);

void nnet_subtract_gradients(NeuralNet *nnet, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float learn_rate, int num_training_examples);

float nnet_cost_function_MSE(float *outputs, float *expected);
float nnet_cost_function_CCE(float *outputs, float *expected);

void nnet_layer_function_dense_relu(float **weights, int rows, int columns, float *activations, float *bias, float *destination);
void nnet_layer_function_dense_sigmoid(float **weights, int rows, int columns, float *activations, float *bias, float *destination);
void nnet_layer_function_dense_softmax(float **weights, int rows, int columns, float *activations, float *bias, float *destination);

void nnet_layer_function_dense_deriv_weights_and_biases(float *current_chain_deriv, int current_layer, float *activation, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS]);
void nnet_layer_function_dense_deriv_activations(NeuralNet* nnet, float* current_chain_deriv, float* math_buffer, int current_layer);
void nnet_activation_function_deriv_relu(float* current_chain_deriv, int current_layer, float* activation);
void nnet_activation_function_deriv_sigmoid(float* current_chain_deriv, int current_layer, float* activation);
void nnet_cost_function_deriv_MSE(float* destination, float *activations, float* training_output);

void nnet_backprop(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float *current_chain_deriv, float *math_buffer, float *training_input, float *training_output);


float sigmoid(float n)
{
    return (1 / (1 + powf(2.71828183F, -n)));
}

float relu(float n)
{
    return n*(n>0);
}

NeuralNet *nnet_init(float init_min, float init_max)
{
    NeuralNet *new_network = (NeuralNet *)malloc(sizeof(NeuralNet));
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        new_network->weights[i] = laa_allocRandMatrix(npl[i + 1], npl[i], init_min, init_max);
        new_network->biases[i] = laa_allocRandVector(npl[i + 1], init_min, init_max);
    }
    return new_network;
}

void nnet_free(NeuralNet *nnet)
{
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        laa_freeMatrix(nnet->weights[i], npl[i + 1]);
        laa_freeVector(nnet->biases[i]);
    }
    free(nnet);
}

void nnet_reset_network(NeuralNet *nnet)
{
    srand(rand() % 0xffffffff);
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        laa_setMatrixToRand(nnet->weights[i], npl[i + 1], npl[i]);
        laa_setVectorToRand(nnet->biases[i], npl[i + 1]);
    }
}

float nnet_cost_function_MSE(float *outputs, float *expected)
{
    float sum = 0;
    for (int i = 0; i < npl[LAYERS]; i++)
        sum += (outputs[i] - expected[i]) * (outputs[i] - expected[i]);
    return sum;
}

float nnet_cost_function_CCE(float *outputs, float *expected) 
{
    float total = 0.0;
    for (int i = 0; i < npl[LAYERS]; i++)
        total += expected[i] * log(outputs[i]);
    return -total;
}

float nnet_total_cost(NeuralNet *nnet, float **inputs, float **outputs, int num_data_points)
{
    float *activations[LAYERS];
    float sum = 0;
    for (int i = 0; i < LAYERS; i++)
    {
        activations[i] = laa_allocVector(npl[i + 1], 0);
    }
    for (int i = 0; i < num_data_points; i++)
    {
        sum += nnet_cost_function_MSE(nnet_feed_forward(inputs[i], nnet, activations), outputs[i]) / num_data_points;
    }
    for (int i = 0; i < LAYERS; i++)
    {
        laa_freeVector(activations[i]);
    }
    return sum;
}

//computes activations of the next layer and outputs it into destination
void nnet_layer_function_dense_relu(float **weights, int rows, int columns, float *activations, float *bias, float *destination)
{
    for (int i = 0; i < rows; i++)
        destination[i] = relu(laa_dot(activations, weights[i], columns) + bias[i]);
}

void nnet_layer_function_dense_sigmoid(float **weights, int rows, int columns, float *activations, float *bias, float *destination)
{
    for (int i = 0; i < rows; i++)
        destination[i] = sigmoid(laa_dot(activations, weights[i], columns) + bias[i]);
}

void nnet_layer_function_dense_softmax(float **weights, int rows, int columns, float *activations, float *bias, float *destination)
{
    float sum = 0;
    for (int i = 0; i < rows; i++)
        sum+=powf(2.718281828459, activations[i]);
    for (int i = 0; i < rows; i++)
        destination[i] = powf(2.718281828459, activations[i])/sum;
}

float *nnet_feed_forward(float *inputs, NeuralNet *nnet, float *activations[LAYERS])
{
    int i;
    //hidden layers:
    nnet_layer_function_dense_relu(nnet->weights[0], npl[1], npl[0], inputs, nnet->biases[0], activations[0]);
    for (i = 1; i < LAYERS-1; i++)
        nnet_layer_function_dense_relu(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);

    //last/output layer:
    nnet_layer_function_dense_relu(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);

    return activations[LAYERS - 1];
}

// calculates the gradient of the weights and biases of the current layer by multiplying their derivatives with respect to the current layer by the 
// current chain-rule product. 
void nnet_layer_function_dense_deriv_weights_and_biases(float *current_chain_deriv, int current_layer, float *activation, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS])
{
    // computes current_chain_deriv * activations' and adds result into weight gradient
    for (int i = 0; i < npl[current_layer+1]; i ++)
    {
        for (int j = 0; j < npl[current_layer]; j++)
        {
            weight_gradients[current_layer][i][j] += current_chain_deriv[i]*activation[j];
        }
        bias_gradients[current_layer][i] += current_chain_deriv[i];
    }
}

// calculates the derivative of the current layer's activations with respect to the last layer's, and multiplies that by the 
// current chain-rule product to update it. 
void nnet_layer_function_dense_deriv_activations(NeuralNet* nnet, float* current_chain_deriv, float* math_buffer, int current_layer)
{
    multiply_MtVreplace(nnet->weights[current_layer], npl[current_layer+1], npl[current_layer], current_chain_deriv, math_buffer);
} 

void nnet_activation_function_deriv_relu(float* current_chain_deriv, int current_layer, float* activation)
{
    for (int i = 0; i < npl[current_layer+1]; i ++)
        current_chain_deriv[i] *= (activation[i] > 0);
}

void nnet_activation_function_deriv_sigmoid(float* current_chain_deriv, int current_layer, float* activation)
{
    float temp;
    for (int i = 0; i < npl[current_layer+1]; i ++)
    {
        temp = sigmoid(activation[i]);
        current_chain_deriv[i] *= temp*(1-temp);
    }
}

//means squared error derivative. Derivative of the cost function with respect to the output of the nnet (activation at last layer)
//places the result in the destination matrix. Column vector with values ai-yi
void nnet_cost_function_deriv_MSE(float* destination, float *activations, float* training_output)
{
    for (int i = 0; i < npl[LAYERS]; i ++)
        destination[i] = activations[i] - training_output[i];
}

// outputs the derivative of the categorical cross entropy loss function with respect to the activations 
// (predictions from the neural net), to the destination array, given the expected predictions (training_output)
void nnet_cost_function_deriv_CCE_softmax(float* destination, float *activations, float* training_output)
{
    for(int i = 0; i < npl[LAYERS]; i++)
        destination[i] = (activations[i] - training_output[i]);
}

void nnet_backprop(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float *current_chain_deriv, float *math_buffer, float *training_input, float *training_output)
{
    int layer = LAYERS-1; 

    nnet_feed_forward(training_input, nnet, activations);

    //set the current chain rule derivative value to the derivative of the cost function with respect to the last activation.
    nnet_cost_function_deriv_MSE(current_chain_deriv, activations[LAYERS-1], training_output);

    for (layer = LAYERS - 1; layer > 0; layer--)
    {
        //applies derivative of activation function (relu, sigmoid, etc) to the current_chain_deriv, in accordance to the chain rule
        nnet_activation_function_deriv_relu(current_chain_deriv, layer, activations[layer]);

        //updates the weight and bias gradients based on the current_chain_deriv.
        nnet_layer_function_dense_deriv_weights_and_biases(current_chain_deriv, layer, activations[layer-1], weight_gradients, bias_gradients);

        //updates the current_chain_deriv matrix for the next layer function derivative.
        nnet_layer_function_dense_deriv_activations(nnet, current_chain_deriv, math_buffer, layer);
    }
    //the first layer
    nnet_activation_function_deriv_relu(current_chain_deriv, layer, activations[layer]);
    nnet_layer_function_dense_deriv_weights_and_biases(current_chain_deriv, layer, training_input, weight_gradients, bias_gradients);
    nnet_layer_function_dense_deriv_activations(nnet, current_chain_deriv, math_buffer, layer);
}

int nnet_optimize(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int iterations, float learn_rate)
{
    //nnet_print(nnet);
    printf("initializing backprop... ");
    const int examples_per_batch = num_mini_batches ? training_set->num_examples / num_mini_batches : 0;
    int iteration;
    int nthExample;
    int batch;
    int i;
    int largest_layer_size = 0;
    float **weight_gradients[LAYERS];
    float *bias_gradients[LAYERS];
    float *activations[LAYERS];
    float *chain_rule_vector;
    float *math_buffer;

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
    chain_rule_vector = laa_allocVector(largest_layer_size, 0);
    math_buffer = laa_allocVectorRaw(largest_layer_size);
    printf("done\n");

    for (iteration = iterations; iteration--;)
    {
        printf("\rtraining... epoch %d/%d", iterations-iteration, iterations);
        //printf("\rtraining... %d/%d cost: %f\n", iterations-iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < num_mini_batches; batch++)
        {
            for (nthExample = batch * examples_per_batch; nthExample < (batch + 1) * examples_per_batch; nthExample++)
            {
                nnet_backprop(nnet, activations, weight_gradients, bias_gradients, chain_rule_vector, math_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
                //break;
            }
            // nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, examples_per_batch);
            // nnet_print(nnet);
            // exit(1);
        }
        for (; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_backprop(nnet, activations, weight_gradients, bias_gradients, chain_rule_vector, math_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, examples_per_batch);

        if (isnan(activations[0][0]))
        {
            printf("\ndiverged!\n");
            return 0;
        }
    }
    printf("done\n");

    //nnet_print(nnet);

    for (i = 0; i < LAYERS; i++)
    {
        laa_freeMatrix(weight_gradients[i], npl[i + 1]);
        laa_freeVector(bias_gradients[i]);
        laa_freeVector(activations[i]);
    }
    laa_freeVector(chain_rule_vector);
    laa_freeVector(math_buffer);
    return 1;
}

int nnet_optimize_parallel(NeuralNet *nnet, TrainingSet *training_set, int parallel_batches, int iterations, float learn_rate)
{
    printf("initializing backprop... ");
    const int examples_per_thread = training_set->num_examples / MAX_THREADS / parallel_batches;
    int iteration;
    int thread;
    int batch;
    int i, j;
    int largest_layer_size = 0;
    float **weight_gradients[MAX_THREADS][LAYERS];
    float *bias_gradients[MAX_THREADS][LAYERS];
    float *activations[MAX_THREADS][LAYERS];
    float *chain_rule_vector[MAX_THREADS];
    float *math_buffer[MAX_THREADS];
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
        chain_rule_vector[i] = laa_allocVectorRaw(largest_layer_size);
        math_buffer[i] = laa_allocVectorRaw(largest_layer_size);
    }
    printf("done\n");

    for (iteration = iterations; iteration--;)
    {
        printf("\rtraining... epoch %d/%d", iterations-iteration, iterations);
        //printf("\rtraining... epoch %d/%d cost: %f", iterations - iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < parallel_batches; batch++)
        {
            #pragma omp parallel for
            for (thread = 0; thread < MAX_THREADS; thread++)
            {
                for (int nthExample = (batch * MAX_THREADS + thread) * examples_per_thread; nthExample < (batch * MAX_THREADS + thread + 1) * examples_per_thread; nthExample++)
                {
                    nnet_backprop(nnet, activations[thread], weight_gradients[thread], bias_gradients[thread], chain_rule_vector[thread], math_buffer[thread], training_set->inputs[nthExample], training_set->outputs[nthExample]);
                }
            }

            for (i = 0; i < MAX_THREADS; i++)
            {
                nnet_subtract_gradients(nnet, weight_gradients[i], bias_gradients[i], learn_rate, examples_per_thread);
            }
        }

        for (int nthExample = parallel_batches * MAX_THREADS * examples_per_thread; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_backprop(nnet, activations[0], weight_gradients[0], bias_gradients[0], chain_rule_vector[0], math_buffer[0], training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients[0], bias_gradients[0], learn_rate, examples_per_thread);

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
        laa_freeVector(chain_rule_vector[i]);
        laa_freeVector(math_buffer[i]);
    }
    printf("done\n");
    return 1;
}

void nnet_subtract_gradients(NeuralNet *nnet, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float learn_rate, int num_training_examples)
{
    int layer, i, j;
    for (layer = 0; layer < LAYERS; layer++)
    {
        for (i = 0; i < npl[layer + 1]; i++)
        {
            for (j = 0; j < npl[layer]; j++)
            {
                nnet->weights[layer][i][j] -= weight_gradients[layer][i][j] * learn_rate / num_training_examples;
                weight_gradients[layer][i][j] = 0;
            }
            nnet->biases[layer][i] -= bias_gradients[layer][i] * learn_rate / num_training_examples;
            bias_gradients[layer][i] = 0;
        }
    }
}

float nnet_test_results(NeuralNet *nnet, TestingSet *test_set, int print_each_test, int print_results)
{
    float *activations[LAYERS];
    int i = 0, numWrong = 0;
    for (i = 0; i < LAYERS; i++)
    {
        activations[i] = laa_allocVector(npl[i + 1], 0);
    }
    for (i = 0; i < test_set->num_examples; i++)
    {
        if (laa_maxIndexValue(test_set->outputs[i], npl[LAYERS]) != laa_maxIndexValue(nnet_feed_forward(test_set->inputs[i], nnet, activations), npl[LAYERS]))
            numWrong++;
    }
    if (print_results)
    {
        printf("\naccuracy: %d/%d (%.3f%%)\n", test_set->num_examples - numWrong, test_set->num_examples, 100 * (test_set->num_examples - numWrong) / (float)test_set->num_examples);
    }
    for (i = 0; i < LAYERS; i++)
    {
        laa_freeVector(activations[i]);
    }
    return (test_set->num_examples - numWrong) / (float)test_set->num_examples;
}

// --- --- --- --- --- --- --- --- File IO  --- --- --- --- --- --- --- ---

void nnet_print(NeuralNet *nnet)
{
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        printf("\nweights:");
        laa_printMatrix(nnet->weights[i], npl[i + 1], npl[i]);
        printf("bias: ");
        laa_printVector(nnet->biases[i], npl[i + 1]);
    }
}

int nnet_save_to_file(NeuralNet *nnet, const char *fileName)
{
    FILE *filePointer = fopen(fileName, "wb");
    if (filePointer == NULL)
        return 0;

    int layers = LAYERS, i;
    fwrite(&layers, sizeof(int), 1, filePointer);
    fwrite(npl, sizeof(int), layers + 1, filePointer);
    for (i = 0; i < LAYERS; i++)
    {
        laa_writeMatrixBin(nnet->weights[i], npl[i + 1], npl[i], filePointer);
        laa_writeVectorBin(nnet->biases[i], npl[i + 1], filePointer);
    }
    fclose(filePointer);
    return 1;
}

int nnet_load_from_file(NeuralNet *nnet, const char *fileName)
{
    FILE *filePointer = fopen(fileName, "rb");
    if (filePointer == NULL)
    {
        return 0;
    }

    int layers, i;
    fread(&layers, sizeof(layers), 1, filePointer);
    int *npl = (int *)malloc(sizeof(int) * (layers + 1));
    fread(npl, sizeof(int), layers + 1, filePointer);

    for (i = 0; i < layers; i++)
    {
        laa_readMatrixBin(nnet->weights[i], filePointer);
        laa_readVectorBin(nnet->biases[i], filePointer);
    }
    fclose(filePointer);
    return 1;
}

//--- --- --- --- --- --- --- --- --- special math functions --- --- --- --- --- --- --- --- --- ---

//untransposed rows and cols
//afterwords the vectorvals should have length of cols
void multiply_MtVreplace(float **matrixVals, int rows, int cols, float *vectorVals, float *buffer)
{
    int i, j;
    float sum = 0;

    for (i = 0; i < rows; i ++)
    {
        buffer[i] = vectorVals[i];
    }

    for (i = 0; i < cols; i++)
    {
        sum = 0;
        for (j = 0; j < rows; j++)
        {
            sum += matrixVals[j][i]*buffer[j];
        }
        vectorVals[i] = sum;
    }
}