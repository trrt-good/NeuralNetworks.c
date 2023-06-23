#include "neural_net_gpu.h"

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float *gpu_weights[LAYERS];
float *gpu_biases[LAYERS];
float *gpu_activations[LAYERS];
float *gpu_training_inputs;
float *gpu_training_outputs;

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

//computes the activation of each neuron in a layer in parallel
//all parameters must be pointing to gpu memory
__global__ void nnet_kernel_layer_function_dense_relu(float *weights, int rows, int columns, float *activations, float *bias, float *destination)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_sum = 0.0;

    if (row < rows)
    {
        for (int i = 0; i < columns; i ++)
        {
            temp_sum += weights[row * columns + i] * activations[i];
        }
        temp_sum += bias[row];
        destination[row] = temp_sum * (temp_sum > 0);
    }
}

//inputs array must be pointing to gpu memory
float* nnet_feed_forward(float *inputs, NeuralNet* nnet, float* activations[LAYERS])
{
    int i;
    //hidden layers:
    dim3 grid_size((MAX_LAYER_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(gpu_weights[0], npl[1], npl[0], inputs, gpu_biases[0], gpu_activations[0]);
    for (i = 1; i < LAYERS-1; i++)
        nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(gpu_weights[i], npl[i + 1], npl[i], gpu_activations[i - 1], gpu_biases[i], gpu_activations[i]);

    //last/output layer:
    nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(gpu_weights[i], npl[i + 1], npl[i], gpu_activations[i - 1], gpu_biases[i], gpu_activations[i]);

    return gpu_activations[LAYERS - 1];
}

// #define LAYERS 3

// typedef struct neural_network
// {
//     float **weights[LAYERS];
//     float *biases[LAYERS];
// } NeuralNet;

// float *gpu_weights[LAYERS];
// float *gpu_biases[LAYERS];
// float *gpu_activations[LAYERS];

void nnet_free_gpu_nnet_and_activations()
{
    for (int i = 0; i < LAYERS; i ++)
    {
        cudaErrorCheck(cudaFree(gpu_activations[i]));
        cudaErrorCheck(cudaFree(gpu_biases[i]));
        cudaErrorCheck(cudaFree(gpu_weights[i]));
    }
}

void nnet_alloc_gpu_nnet_and_activations()
{
    for (int i = 0; i < LAYERS; i ++)
    {
        int layer_size = sizeof(float) * npl[i+1];
        cudaErrorCheck(cudaMalloc((void **)&gpu_weights[i], layer_size * npl[i]));
        cudaErrorCheck(cudaMalloc((void **)&gpu_biases[i], layer_size));
        cudaErrorCheck(cudaMalloc((void **)&gpu_activations[i], layer_size));
    }
}

void nnet_load_nnet_to_GPU(NeuralNet *nnet)
{
    nnet_free_gpu_nnet_and_activations();
    nnet_alloc_gpu_nnet_and_activations();
    for (int i = 0; i < LAYERS; i++)
    {
        for (int j = 0; j < npl[i+1]; j ++)
            cudaErrorCheck(cudaMemcpy((void *)(gpu_weights[i] + j * npl[i]), nnet->weights[i][j], npl[i] * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(gpu_biases[i], nnet->biases[i], npl[i+1] * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void nnet_load_nnet_from_GPU(NeuralNet *nnet)
{
    for (int i = 0; i < LAYERS; i++)
    {
        for (int j = 0; j < npl[i+1]; j ++)
            cudaErrorCheck(cudaMemcpy(nnet->weights[i][j], (void *)(gpu_weights[i] + j * npl[i]), npl[i] * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(nnet->biases[i], gpu_biases[i], npl[i+1] * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void nnet_load_data_to_GPU(TrainingSet *training_data)
{
    nnet_free_gpu_data();

    // Allocate memory on the device
    cudaErrorCheck(cudaMalloc((void **)&gpu_training_inputs, INPUT_LAYER_SIZE * training_data->num_examples * sizeof(float)));
    cudaErrorCheck(cudaMalloc((void **)&gpu_training_outputs, OUTPUT_LAYER_SIZE * training_data->num_examples * sizeof(float)));

    // Copy the data from host to device
    for (int i = 0; i < training_data->num_examples; i++) 
    {
        cudaErrorCheck(cudaMemcpy(gpu_training_inputs + i * INPUT_LAYER_SIZE, training_data->inputs[i], INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(gpu_training_outputs + i * OUTPUT_LAYER_SIZE, training_data->outputs[i], OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void nnet_free_gpu_data()
{
    cudaErrorCheck(cudaFree(gpu_training_inputs));
    cudaErrorCheck(cudaFree(gpu_training_outputs));
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
            }
            nnet_subtract_gradients(nnet, weight_gradients, bias_gradients, learn_rate, examples_per_batch);
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