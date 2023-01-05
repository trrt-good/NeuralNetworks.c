#include "neural_net.h"

#define RELU(a) ((a < 0) ? 0 : 1)

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
void computeOneActivation(float **matrix, int rows, int columns, float *multVect, float *addVect, float *destination);
void multiply_replace(float **a_matrixVals, int a_rows, int a_columns, float **b_matrixVals, int b_rows, int b_columns, float *buffer);
void update_last_two_layer_gradients(NeuralNet *nnet, float *expeced_output);

NeuralNet *nnet_init()
{
    NeuralNet *new_network = malloc(sizeof(NeuralNet));
    int i;
    int largest_layer_size = 0;

    for (i = 0; i < LAYERS; i++)
    {
        if (npl[i] > largest_layer_size)
            largest_layer_size = npl[i];
        new_network->weights[i] = laa_allocRandMatrix(npl[i + 1], npl[i]);
        new_network->weight_gradients[i] = laa_allocMatrix(npl[i + 1], npl[i], 0);

        new_network->biases[i] = laa_allocRandVector(npl[i + 1]);
        new_network->bias_gradients[i] = laa_allocVector(npl[i + 1], 0);

        new_network->activations[i] = laa_allocVector(npl[i + 1], 0);
    }
    if (npl[i] > largest_layer_size)
        largest_layer_size = npl[i];

    new_network->weight_product = laa_allocMatrixRaw(largest_layer_size, largest_layer_size);
    new_network->weight_product_buffer = laa_allocVectorRaw(largest_layer_size);
    return new_network;
}

TestingSet *nnet_testing_set_init(int num_testing_examples)
{
    TestingSet *new_set = malloc(sizeof(TestingSet));
    new_set->num_examples = num_testing_examples;
    int i;
    new_set->inputs = laa_allocMatrixRaw(num_testing_examples, INPUT_LAYER_SIZE);
    new_set->outputs = laa_allocMatrixRaw(num_testing_examples, OUTPUT_LAYER_SIZE);
    return new_set;
}

TrainingSet *nnet_training_set_init(int num_training_examples)
{
    TrainingSet *new_set = malloc(sizeof(TrainingSet));
    new_set->num_examples = num_training_examples;
    int i;
    new_set->inputs = laa_allocMatrixRaw(num_training_examples, INPUT_LAYER_SIZE);
    new_set->outputs = laa_allocMatrixRaw(num_training_examples, OUTPUT_LAYER_SIZE);
    return new_set;
}

void nnet_free(NeuralNet *nnet)
{
    int i;
    laa_freeVector(nnet->weight_product_buffer);
    int max_layer_size = 0;
    for (i = 0; i <= LAYERS; i++)
        if (npl[i] > max_layer_size)
            max_layer_size = npl[i];
    laa_freeMatrix(nnet->weight_product, max_layer_size);
    for (i = 0; i < LAYERS; i++)
    {
        laa_freeMatrix(nnet->weights[i], npl[i + 1]);
        laa_freeMatrix(nnet->weight_gradients[i], npl[i + 1]);

        laa_freeVector(nnet->biases[i]);
        laa_freeVector(nnet->bias_gradients[i]);

        laa_freeVector(nnet->activations[i]);
    }
    free(nnet);
}

void nnet_free_test_set(TestingSet *set)
{
    laa_freeMatrix(set->inputs, set->num_examples);
    laa_freeMatrix(set->outputs, set->num_examples);
    free(set);
}

void nnet_free_training_set(TrainingSet *set)
{
    laa_freeMatrix(set->inputs, set->num_examples);
    laa_freeMatrix(set->outputs, set->num_examples);
    free(set);
}

void nnet_reset_network(NeuralNet *nnet)
{
    srand(rand() % 0xffffffff);
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        laa_setMatrixToRand(nnet->weights[i], npl[i + 1], npl[i]);
        laa_setMatrixTo(nnet->weight_gradients[i], npl[i + 1], npl[i], 0);

        laa_setVectorToRand(nnet->biases[i], npl[i + 1]);
        laa_setVectorTo(nnet->bias_gradients[i], npl[i + 1], 0);

        laa_setVectorTo(nnet->activations[i], npl[i + 1], 0);
    }
}

void nnet_load_data(TrainingSet *training_set, TestingSet *testing_set, char *fileName, char *delimiter, int bufferSize)
{
    readRowData_ML(fileName, delimiter, bufferSize, training_set->num_examples + testing_set->num_examples, testing_set->num_examples, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, training_set->inputs, training_set->outputs, testing_set->inputs, testing_set->outputs, 1);
    //     laa_printMatrix(training_set->inputs, training_set->num_examples, npl[0]);
    //     laa_printMatrix(training_set->outputs, training_set->num_examples, npl[LAYERS]);
    //     laa_printMatrix(testing_set->inputs, testing_set->num_examples, npl[0]);
    //     laa_printMatrix(testing_set->outputs, testing_set->num_examples, npl[LAYERS]);
}

float *nnet_run_data(float inputs[npl[0]], NeuralNet *nnet)
{
    computeOneActivation(nnet->weights[0], npl[1], npl[0], inputs, nnet->biases[0], nnet->activations[0]);
    int i;
    for (i = 1; i < LAYERS; i++)
    {
        computeOneActivation(nnet->weights[i], npl[i + 1], npl[i], nnet->activations[i - 1], nnet->biases[i], nnet->activations[i]);
    }
    return nnet->activations[LAYERS - 1];
}

float nnet_total_cost(NeuralNet *nnet, TrainingSet *set)
{
    float sum = 0;
    int i, j;
    for (i = 0; i < set->num_examples; i++)
    {
        nnet_run_data(set->inputs[i], nnet);
        for (j = 0; j < npl[LAYERS]; j++)
        {
            sum += (set->outputs[i][j] - nnet->activations[LAYERS - 1][j]) * (set->outputs[i][j] - nnet->activations[LAYERS - 1][j]) / set->num_examples;
        }
    }
    return sum;
}

int nnet_iterate_gradients(NeuralNet *nnet, float *training_input, float *training_output)
{
    int layer;
    int i, j, k, l;
    float dotSum1 = 0;
    float dotSum2 = 0;

    nnet_run_data(training_input, nnet);

    // compute the last layer two gradients so that the rest can be calculated in a simple algorithm
    update_last_two_layer_gradients(nnet, training_output);

    // set the weight product to the last layer's weight
    laa_copyMatrixValues(nnet->weights[LAYERS - 1], nnet->weight_product, npl[LAYERS], npl[LAYERS - 1]);

    for (layer = LAYERS - 2; layer > 1; layer--)
    {
        multiply_replace(nnet->weight_product, npl[layer + 2], npl[layer + 1], nnet->weights[layer], npl[layer + 1], npl[layer], nnet->weight_product_buffer);
        for (i = 0; i < npl[layer]; i++)
        {
            for (j = 0; j < npl[layer - 1]; j++)
            {
                dotSum1 = 0;
                for (k = 0; k < npl[LAYERS]; k++)
                {
                    dotSum1 += nnet->weight_product[k][i] * nnet->activations[layer - 1][j] * (nnet->activations[LAYERS - 1][k] - training_output[k]);
                }
                nnet->weight_gradients[layer - 1][i][j] += dotSum1;
            }

            dotSum2 = 0;
            for (k = 0; k < npl[LAYERS]; k++)
            {
                dotSum2 += nnet->weight_product[k][i] * (nnet->activations[LAYERS - 1][k] - training_output[k]);
            }
            nnet->bias_gradients[layer - 1][i] += dotSum2;
        }
    }
    multiply_replace(nnet->weight_product, npl[layer + 2], npl[layer + 1], nnet->weights[layer], npl[layer + 1], npl[layer], nnet->weight_product_buffer);
    for (i = 0; i < npl[layer]; i++)
    {
        for (j = 0; j < npl[layer - 1]; j++)
        {
            dotSum1 = 0;
            for (k = 0; k < npl[LAYERS]; k++)
            {
                dotSum1 += nnet->weight_product[k][i] * training_input[j] * (nnet->activations[LAYERS - 1][k] - training_output[k]);
            }
            nnet->weight_gradients[layer - 1][i][j] += dotSum1;
        }

        dotSum2 = 0;
        for (k = 0; k < npl[LAYERS]; k++)
        {
            dotSum2 += nnet->weight_product[k][i] * (nnet->activations[LAYERS - 1][k] - training_output[k]);
        }
        nnet->bias_gradients[layer - 1][i] += dotSum2;
    }
}

int nnet_backprop(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int iterations, float learn_rate)
{
    const int examples_per_batch = num_mini_batches ? training_set->num_examples / num_mini_batches : 0;
    int iteration;
    int nthExample;
    int layer;
    int batch;
    for (iteration = iterations; iteration--;)
    {
        for (batch = 0; batch < num_mini_batches; batch++)
        {
            for (nthExample = batch * examples_per_batch; nthExample < (batch + 1) * examples_per_batch; nthExample++)
            {
                nnet_iterate_gradients(nnet, training_set->inputs[nthExample], training_set->outputs[nthExample]);
            }
            nnet_subtract_gradients(nnet, learn_rate, training_set->num_examples);
        }
        for (; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_iterate_gradients(nnet, training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, learn_rate, training_set->num_examples);

        if (isnan(nnet->activations[0][0]))
        {
            return 0;
        }
    }
    return 1;
}

void update_last_two_layer_gradients(NeuralNet *nnet, float *expected_output)
{
    int i, j, k;
    float diff, sum;
    for (i = 0; i < npl[LAYERS]; i++)
    {
        diff = nnet->activations[LAYERS - 1][i] - expected_output[i];
        nnet->bias_gradients[LAYERS - 1][i] += diff;
        for (j = 0; j < npl[LAYERS - 1]; j++)
        {
            nnet->weight_gradients[LAYERS - 1][i][j] += diff * nnet->activations[LAYERS - 2][j];

            // because it's += this essentially acts like a dot product for each row in the
            //  bias gradient, completing the necessary vector-matrix multiplcation while still using
            //  the same loops.
            nnet->bias_gradients[LAYERS - 2][j] += nnet->weights[LAYERS - 1][i][j] * diff;

            // dot product:
            for (k = 0; k < npl[LAYERS - 2]; k++)
            {
                nnet->weight_gradients[LAYERS - 2][j][k] += nnet->weights[LAYERS - 1][i][j] * diff * nnet->activations[LAYERS - 3][k];
            }
        }
    }
}

void nnet_subtract_gradients(NeuralNet *nnet, float learn_rate, int num_training_examples)
{
    //(RELU(activations[layer][i]))
    int layer, i, j;
    for (layer = 0; layer < LAYERS; layer++)
    {
        for (i = 0; i < npl[layer + 1]; i++)
        {
            for (j = 0; j < npl[layer]; j++)
            {
                nnet->weights[layer][i][j] -= nnet->weight_gradients[layer][i][j] * learn_rate / num_training_examples;
                nnet->weight_gradients[layer][i][j] = 0;
            }
            nnet->biases[layer][i] -= nnet->bias_gradients[layer][i] * learn_rate / num_training_examples;
            nnet->bias_gradients[layer][i] = 0;
        }
    }
}

float nnet_test_results(NeuralNet *nnet, TestingSet *test_set, int print_each_test, int print_results)
{
    int i = 0, numWrong = 0;
    for (i = 0; i < test_set->num_examples; i++)
    {
        if (laa_maxIndexValue(test_set->outputs[i], npl[LAYERS]) != laa_maxIndexValue(nnet_run_data(test_set->inputs[i], nnet), npl[LAYERS]))
            numWrong++;

        if (print_each_test)
        {
            printf("\ndesired output: ");
            laa_printVector(test_set->outputs[i], npl[LAYERS]);
            printf("predicted output: ");
            laa_printVector(nnet->activations[LAYERS - 1], npl[LAYERS]);
        }
    }
    if (print_results)
    {
        printf("\naccuracy: %d/%d (%.3f%%)\n", test_set->num_examples - numWrong, test_set->num_examples, 100 * (test_set->num_examples - numWrong) / (float)test_set->num_examples);
    }
    return (test_set->num_examples - numWrong) / (float)test_set->num_examples;
}

// --- --- --- --- --- --- --- --- File IO  --- --- --- --- --- --- --- ---
#pragma region FileIO

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
    int *npl = malloc(sizeof(int) * (layers + 1));
    fread(npl, sizeof(int), layers + 1, filePointer);

    for (i = 0; i < layers; i++)
    {
        laa_readMatrixBin(nnet->weights[i], filePointer);
        laa_readVectorBin(nnet->biases[i], filePointer);
    }
    fclose(filePointer);
}
#pragma endregion

//--- --- --- --- --- --- --- --- --- special math functions --- --- --- --- --- --- --- --- --- ---

/**
 * @brief preforms operation matrix*multVect + addVect and
 * stores the result in destination
 *
 * @param matrix
 * @param rows
 * @param columns
 * @param multVect
 * @param addVect
 * @param destination
 */
void computeOneActivation(float **weights, int rows, int columns, float *preLayer, float *bias, float *destination)
{
    int i;
    for (i = 0; i < rows; i++)
    {
        // destination[i] = max(0, laa_dot(preLayer, weights[i], columns) + bias[i]);
        destination[i] = laa_dot(preLayer, weights[i], columns) + bias[i];
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