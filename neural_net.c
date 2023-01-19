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
void computeOneActivation(float **matrix, int rows, int columns, float *multVect, float *addVect, float *destination);
void multiply_replace(float **a_matrixVals, int a_rows, int a_columns, float **b_matrixVals, int b_rows, int b_columns, float *buffer);
void update_last_two_layer_gradients(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float *expected_output);

float sigmoid(float n) {
    return (1 / (1 + powf(2.71828183F, -n)));
}

NeuralNet *nnet_init(float init_min, float init_max)
{
    NeuralNet *new_network = malloc(sizeof(NeuralNet));
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        new_network->weights[i] = laa_allocRandMatrix(npl[i + 1], npl[i], init_min, init_max);
        new_network->biases[i] = laa_allocRandVector(npl[i + 1], init_min, init_max);
    }
    return new_network;
}

TestingSet *nnet_testing_set_init(int num_testing_examples)
{
    TestingSet *new_set = malloc(sizeof(TestingSet));
    new_set->num_examples = num_testing_examples;
    int i;
    new_set->inputs = laa_allocMatrix(num_testing_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_testing_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
}

TrainingSet *nnet_training_set_init(int num_training_examples)
{
    TrainingSet *new_set = malloc(sizeof(TrainingSet));
    new_set->num_examples = num_training_examples;
    int i;
    new_set->inputs = laa_allocMatrix(num_training_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_training_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
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
        laa_setVectorToRand(nnet->biases[i], npl[i + 1]);
    }
}

void nnet_load_data(TrainingSet *training_set, TestingSet *testing_set)
{
    printf("loading data... ");
    // read_iris_data("Data/iris.txt", ",", 64, training_set->num_examples + testing_set->num_examples, testing_set->num_examples, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, training_set->inputs, training_set->outputs, testing_set->inputs, testing_set->outputs, 1);
    read_mnist_number_data("../mnist/mnist_train.csv", training_set->num_examples, training_set->inputs, training_set->outputs);
    read_mnist_number_data("../mnist/mnist_test.csv", testing_set->num_examples, testing_set->inputs, testing_set->outputs);

        // laa_printMatrix(training_set->inputs, training_set->num_examples, npl[0]);
        // laa_printMatrix(training_set->outputs, training_set->num_examples, npl[LAYERS]);
        // laa_printMatrix(testing_set->inputs, testing_set->num_examples, npl[0]);
        // laa_printMatrix(testing_set->outputs, testing_set->num_examples, npl[LAYERS]);
    printf("done\n");
}

void nnet_shuffle_data(float **inputs, float **outputs, int n)
{
    srand(time(NULL));
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        float *tmp1 = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = tmp1;
        float *tmp2 = outputs[i];
        outputs[i] = outputs[j];
        outputs[j] = tmp2;
    }
}

float cost(float* outputs, float* expected)
{
    float sum;
    for (int i = 0; i < npl[LAYERS]; i ++)
    {
        sum += (outputs[i] - expected[i])*(outputs[i] - expected[i]);
    }
    return sum;
}

float nnet_total_cost(NeuralNet* nnet, float** inputs, float** outputs, int n)
{
    float *activations[LAYERS];
    float sum = 0;
    for (int i = 0; i < LAYERS; i++)
    {
        activations[i] = laa_allocVector(npl[i + 1], 0);
    }
    for (int i = 0; i < n; i ++)
    {
        sum += cost(nnet_run_data(inputs[i], nnet, activations), outputs[i])/n;
    }
    laa_printVector(activations[LAYERS-1], npl[LAYERS]);
    for (int i = 0; i < LAYERS; i++)
    {
        laa_freeVector(activations[i]);
    }
    return sum;
}

float *nnet_run_data(float inputs[npl[0]], NeuralNet *nnet, float *activations[LAYERS])
{
    computeOneActivation(nnet->weights[0], npl[1], npl[0], inputs, nnet->biases[0], activations[0]);
    int i;
    // #pragma omp parallel for
    for (i = 1; i < LAYERS; i++)
    {
        computeOneActivation(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);
    }
    return activations[LAYERS - 1];
}

int nnet_iterate_gradients(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float **weight_product, float *weight_product_buffer, float *training_input, float *training_output)
{
    int layer;
    int i, j, k, l;
    float dotSum1 = 0;
    float dotSum2 = 0;

    nnet_run_data(training_input, nnet, activations);

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
                weight_gradients[layer - 1][i][j] += dotSum1*ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
            }

            dotSum2 = 0;
            for (k = 0; k < npl[LAYERS]; k++)
            {
                dotSum2 += weight_product[k][i] * (activations[LAYERS - 1][k] - training_output[k]);
            }
            bias_gradients[layer - 1][i] += dotSum2*ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
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
            weight_gradients[layer - 1][i][j] += dotSum1*ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
        }

        dotSum2 = 0;
        for (k = 0; k < npl[LAYERS]; k++)
        {
            dotSum2 += weight_product[k][i] * (activations[LAYERS - 1][k] - training_output[k]);
        }
        bias_gradients[layer - 1][i] += dotSum2*ACTIVATION_FUNCTION_DERIV(activations[layer - 1][i]);
    }
}

int nnet_backprop(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int iterations, float learn_rate)
{
    printf("initializing backprop... ");
    const int examples_per_batch = num_mini_batches ? training_set->num_examples / num_mini_batches : 0;
    int iteration;
    int nthExample;
    int layer;
    int batch;
    int i;
    int largest_layer_size;
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
        //printf("\rtraining... %d/%d cost: %f", iterations-iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < num_mini_batches; batch++)
        {
            for (nthExample = batch * examples_per_batch; nthExample < (batch + 1) * examples_per_batch; nthExample++)
            {
                nnet_iterate_gradients(nnet, activations, weight_gradients, bias_gradients, weight_product, weight_product_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
            }
            nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, training_set->num_examples);
        }
        for (; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_iterate_gradients(nnet, activations, weight_gradients, bias_gradients, weight_product, weight_product_buffer, training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, training_set->num_examples);

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

int nnet_backprop_parallel(NeuralNet *nnet, TrainingSet *training_set, int parallel_batches, int iterations, float learn_rate)
{
    printf("initializing backprop... ");
    const int examples_per_thread = training_set->num_examples / MAX_THREADS / parallel_batches ;
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
        //printf("\r%d/%d", iterations-iteration, iterations);
        //printf("\rtraining... %d/%d cost: %f", iterations-iteration, iterations, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < parallel_batches; batch ++)
        {
            #pragma omp parallel for
            for (thread = 0; thread < MAX_THREADS; thread++)
            {
                for (int nthExample = (batch * MAX_THREADS + thread) * examples_per_thread; nthExample < (batch * MAX_THREADS + thread + 1) * examples_per_thread; nthExample++)
                {
                    nnet_iterate_gradients(nnet, activations[thread], weight_gradients[thread], bias_gradients[thread], weight_product[thread], weight_product_buffer[thread], training_set->inputs[nthExample], training_set->outputs[nthExample]);
                }
            }

            for (i = 0; i < MAX_THREADS; i++)
            {
                nnet_subtract_gradients(nnet, weight_gradients[i], bias_gradients[i], learn_rate, training_set->num_examples);
            }
        }

        for (int nthExample = parallel_batches * MAX_THREADS * examples_per_thread; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_iterate_gradients(nnet, activations[0], weight_gradients[0], bias_gradients[0], weight_product[0], weight_product_buffer[0], training_set->inputs[nthExample], training_set->outputs[nthExample]);
        }
        nnet_subtract_gradients(nnet, weight_gradients[0], bias_gradients[0], learn_rate, training_set->num_examples);

        if (isnan(activations[LAYERS-1][0][0]))
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

void update_last_two_layer_gradients(NeuralNet *nnet, float *activations[LAYERS], float **weight_gradients[LAYERS], float *bias_gradients[LAYERS], float *expected_output)
{
    int i, j, k;
    float diff, sum, dotSum;
    for (i = 0; i < npl[LAYERS]; i++)
    {
        diff = activations[LAYERS - 1][i] - expected_output[i];
        bias_gradients[LAYERS - 1][i] += diff*ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 1][i]);
        //printf("b%d[%d]: %f\n", LAYERS - 1, i, bias_gradients[LAYERS - 1][i]);
        for (j = 0; j < npl[LAYERS - 1]; j++)
        {
            weight_gradients[LAYERS - 1][i][j] += diff * activations[LAYERS - 2][j]* ACTIVATION_FUNCTION_DERIV(activations[LAYERS - 1][i]);

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
        if (laa_maxIndexValue(test_set->outputs[i], npl[LAYERS]) != laa_maxIndexValue(nnet_run_data(test_set->inputs[i], nnet, activations), npl[LAYERS]))
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