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

void nnet_subtract_gradients(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float learn_rate, int batch_size);

float nnet_cost_function_MSE(float *outputs, float *expected);
float nnet_cost_function_CCE(float *outputs, float *expected);

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

void nnet_free_gpu_wba(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float *d_activations[LAYERS])
{
    for (int i = 0; i < LAYERS; i ++)
    {
        cudaErrorCheck(cudaFree(d_activations[i]));
        cudaErrorCheck(cudaFree(d_biases[i]));
        cudaErrorCheck(cudaFree(d_bias_gradients[i]));
        cudaErrorCheck(cudaFree(d_weights[i]));
        cudaErrorCheck(cudaFree(d_weight_gradients[i]));
    }
}

void nnet_alloc_gpu_wba(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float *d_activations[LAYERS])
{
    for (int i = 0; i < LAYERS; i ++)
    {
        int layer_size = sizeof(float) * npl[i+1];
        cudaErrorCheck(cudaMalloc((void **)&d_weights[i], layer_size * npl[i]));
        cudaErrorCheck(cudaMalloc((void **)&d_weight_gradients[i], layer_size * npl[i]));
        cudaErrorCheck(cudaMalloc((void **)&d_biases[i], layer_size));
        cudaErrorCheck(cudaMalloc((void **)&d_bias_gradients[i], layer_size));
        cudaErrorCheck(cudaMalloc((void **)&d_activations[i], layer_size));
    }
}

void nnet_alloc_gpu_data(float *d_training_inputs, float *d_training_outputs, int num_examples)
{
    cudaErrorCheck(cudaMalloc((void **)&d_training_inputs, INPUT_LAYER_SIZE * num_examples * sizeof(float)));
    cudaErrorCheck(cudaMalloc((void **)&d_training_outputs, OUTPUT_LAYER_SIZE * num_examples * sizeof(float)));
}

void nnet_free_gpu_data(float *d_training_inputs, float *d_training_outputs)
{
    cudaErrorCheck(cudaFree(d_training_inputs));
    cudaErrorCheck(cudaFree(d_training_outputs));
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

//cpu method. Predictions and outputs should be on cpu RAM
float nnet_total_cost(float **correct_outputs, float ** predictions, int num_data_points)
{
    float sum = 0;
    for (int i = 0; i < num_data_points; i++)
    {
        sum += nnet_cost_function_MSE(correct_outputs[i], predictions[i]) / num_data_points;
    }
    return sum;
}

//computes the activation of each neuron in a layer in parallel
//all parameters must be pointing to gpu memory
__global__ void nnet_kernel_layer_function_dense_relu(float *d_weights, int rows, int columns, float *d_activations, float *d_bias, float *d_destination)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    float temp_sum = 0.0;

    if (row < rows)
    {
        for (int i = 0; i < columns; i ++)
        {
            temp_sum += d_weights[row * columns + i] * d_activations[i];
        }
        temp_sum += d_bias[row];
        d_destination[row] = temp_sum * (temp_sum > 0);
    }
}

//inputs array must be pointing to gpu memory
void nnet_feed_forward(float *d_inputs, float *d_weights[LAYERS], float *d_biases[LAYERS], float *d_activations[LAYERS])
{
    int i;
    //hidden layers:
    dim3 grid_size((MAX_LAYER_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(d_weights[0], npl[1], npl[0], d_inputs, d_biases[0], d_activations[0]);
    for (i = 1; i < LAYERS-1; i++)
        nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(d_weights[i], npl[i + 1], npl[i], d_activations[i - 1], d_biases[i], d_activations[i]);

    //last/output layer:
    nnet_kernel_layer_function_dense_relu<<<grid_size, block_size>>>(d_weights[i], npl[i + 1], npl[i], d_activations[i - 1], d_biases[i], d_activations[i]);
}

__global__ void nnnet_kernel_layer_function_dense_deriv_weights_and_biases(float *d_current_chain_deriv, int current_layer, float *d_activation, float *d_weight_gradient, float *d_bias_gradient)
{
    const int d_npl[LAYERS + 1] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j;
    if (i < d_npl[current_layer+1])
    {
        for (j = 0; j < d_npl[current_layer]; j ++)
        {
            d_weight_gradient[i*d_npl[current_layer] + j] += d_current_chain_deriv[i]*d_activation[j];
        }
        d_bias_gradient[i] += d_current_chain_deriv[i];
    }
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

__global__ void nnet_kernel_layer_function_dense_deriv_activations(float *d_weights, int current_layer, float *d_current_chain_deriv)
{
    const int d_npl[LAYERS + 1] = {INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZES, OUTPUT_LAYER_SIZE};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int temp[MAX_LAYER_SIZE];
    float sum = 0;
    int j;

    if (i < d_npl[current_layer + 1])
        temp[i] = d_current_chain_deriv[i];
    __syncthreads();
    if (i < d_npl[current_layer])
    {
        for (j = 0; j < d_npl[current_layer + 1]; j++)
        {
            sum += d_weights[j * d_npl[current_layer] + i] * temp[i];
        }
        d_current_chain_deriv[i] = sum;
    }
}

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

// calculates the derivative of the current layer's activations with respect to the last layer's, and multiplies that by the 
// current chain-rule product to update it. 
void nnet_layer_function_dense_deriv_activations(NeuralNet* nnet, float* current_chain_deriv, float* math_buffer, int current_layer)
{
    multiply_MtVreplace(nnet->weights[current_layer], npl[current_layer+1], npl[current_layer], current_chain_deriv, math_buffer);
} 

//layer_size should be equal to npl[current_layer+1]
__global__ void nnet_kernel_activation_function_deriv_relu(float *d_current_chain_deriv, int layer_size, float *d_activation)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < layer_size)
        d_current_chain_deriv[i] *= (d_activation[i] > 0);
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

//layer_size is equal to npl[LAYERS]
__global__ void nnet_kernel_cost_function_deriv_MSE(float *d_destination, float *d_activations, float *d_training_output, int layer_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < layer_size)
        d_destination[i] = d_activations[i] - d_training_output[i];
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

void nnet_backprop(float *d_weights[LAYERS], float *d_biases[LAYERS], float *d_activations[LAYERS], float *d_weight_gradients[LAYERS], float *d_bias_gradients[LAYERS], float *d_current_chain_deriv, float *d_training_input, float *d_training_output)
{
    int layer = LAYERS-1; 

    nnet_feed_forward(d_training_input, d_weights, d_biases, d_activations);

    dim3 grid_size((MAX_LAYER_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    
    //set the current chain rule derivative value to the derivative of the cost function with respect to the last activation.
    nnet_kernel_cost_function_deriv_MSE<<<grid_size, block_size>>>(d_current_chain_deriv, d_activations[LAYERS-1], d_training_output, npl[LAYERS]);

    for (layer = LAYERS - 1; layer > 0; layer--)
    {
        //applies derivative of activation function (relu, sigmoid, etc) to the current_chain_deriv, in accordance to the chain rule
        nnet_kernel_activation_function_deriv_relu<<<grid_size, block_size>>>(d_current_chain_deriv, npl[layer+1], d_activations[layer]);

        //updates the weight and bias gradients based on the current_chain_deriv.
        nnnet_kernel_layer_function_dense_deriv_weights_and_biases<<<grid_size, block_size>>>(d_current_chain_deriv, layer, d_activations[layer-1], d_weight_gradients[layer], d_bias_gradients[layer]);

        //updates the current_chain_deriv matrix for the next layer function derivative.
        nnet_kernel_layer_function_dense_deriv_activations<<<grid_size, block_size>>>(d_weights[layer], layer, d_current_chain_deriv);
    }
    //the first layer
    nnet_kernel_activation_function_deriv_relu<<<grid_size, block_size>>>(d_current_chain_deriv, npl[layer+1], d_activations[layer]);
    nnnet_kernel_layer_function_dense_deriv_weights_and_biases<<<grid_size, block_size>>>(d_current_chain_deriv, layer, d_training_input, d_weight_gradients[layer], d_bias_gradients[layer]);
    nnet_kernel_layer_function_dense_deriv_activations<<<grid_size, block_size>>>(d_weights[layer], layer, d_current_chain_deriv);
}

__global__ void nnet_kernel_subtract_gradients(float *d_weights, float *d_weight_gradients, float *d_biases, float *d_bias_gradients, float multiplier, int rows, int cols)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows)
    {
        for (int j = 0; j < cols; j ++)
        {
            d_weights[i * cols + j] -= d_weight_gradients[i * cols + j] * multiplier;
            d_weight_gradients[i * cols + j] = 0;
        }
        d_biases[i] -= d_bias_gradients[i] * multiplier;
        d_bias_gradients[i] = 0;
    }
}

__host__ void nnet_subtract_gradients(float *d_weights[LAYERS], float *d_weight_gradients[LAYERS], float *d_biases[LAYERS], float *d_bias_gradients[LAYERS], float learn_rate, int batch_size)
{
    int layer;
    float mult = learn_rate / batch_size;

    dim3 grid_size((MAX_LAYER_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);

    for (layer = 0; layer < LAYERS; layer++)
    {
        nnet_kernel_subtract_gradients<<<grid_size, block_size>>>(d_weights[layer], d_weight_gradients[layer], d_biases[layer], d_bias_gradients[layer], mult, npl[layer+1], npl[layer]);
    }
}

__global__ void test_kernel(float *d_a, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        d_a[i] = 10.0f;
}

int nnet_optimize(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int epochs, float learn_rate)
{
    //nnet_print(nnet);
    printf("initializing backprop... ");
    const int examples_per_batch = num_mini_batches ? training_set->num_examples / num_mini_batches : 0;
    int epoch, batch, nthExample;
    int i, j;
    int largest_layer_size = 0;

    // allocation of host data
    float *weight_gradients[LAYERS];
    float *bias_gradients[LAYERS];
    float *activations[LAYERS];
    float *chain_rule_vector;

    // initialization of host data
    for (i = 0; i <= LAYERS; i ++)
        if (npl[i] > largest_layer_size)
            largest_layer_size = npl[i];

    for (i = 0; i < LAYERS; i++)
    {
        weight_gradients[i] = laa_allocVector(npl[i + 1] * npl[i], 0);
        bias_gradients[i] = laa_allocVector(npl[i + 1], 0);
        activations[i] = laa_allocVector(npl[i + 1], 0);
    }
    chain_rule_vector = laa_allocVector(largest_layer_size, 0);

    // allocation of device data
    float *d_weight_gradients[LAYERS];
    float *d_bias_gradients[LAYERS];

    float *d_weights[LAYERS];
    float *d_biases[LAYERS];
    float *d_activations[LAYERS]; //activations don't need to be copied from host to device because their values will be set in the first forward pass

    float *d_chain_rule_vector;

    float *d_training_inputs;
    float *d_training_outputs;

    //nnet_alloc_gpu_wba(d_weights, d_weight_gradients, d_biases, d_bias_gradients, d_activations);
    //nnet_alloc_gpu_data(d_training_inputs, d_training_outputs, training_set->num_examples);

    for (i = 0; i < LAYERS; i ++)
    {
        int layer_size = sizeof(float) * npl[i+1];
        cudaErrorCheck(cudaMalloc((void **)&d_weights[i], layer_size * npl[i]));
        cudaErrorCheck(cudaMalloc((void **)&d_weight_gradients[i], layer_size * npl[i]));
        cudaErrorCheck(cudaMalloc((void **)&d_biases[i], layer_size));
        cudaErrorCheck(cudaMalloc((void **)&d_bias_gradients[i], layer_size));
        cudaErrorCheck(cudaMalloc((void **)&d_activations[i], layer_size));
    }

    cudaErrorCheck(cudaMalloc((void **)&d_training_inputs, INPUT_LAYER_SIZE * training_set->num_examples * sizeof(float)));
    cudaErrorCheck(cudaMalloc((void **)&d_training_outputs, OUTPUT_LAYER_SIZE * training_set->num_examples * sizeof(float)));

    cudaMalloc((void **)&d_chain_rule_vector, largest_layer_size * sizeof(float));

    // initialization of device data (copying from host)
    for (i = 0; i < LAYERS; i++)
    {
        // Copy the weight and bias gradients from host to device
        cudaMemcpy(d_weight_gradients[i], weight_gradients[i], npl[i] * npl[i+1] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_gradients[i], bias_gradients[i], npl[i+1] * sizeof(float), cudaMemcpyHostToDevice);

        // Copy the weights and biases from host neural network struct to device
        for (j = 0; j < npl[i+1]; j ++)
            cudaErrorCheck(cudaMemcpy((void *)(d_weights[i] + j * npl[i]), nnet->weights[i][j], npl[i] * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy((void *)d_biases[i], nnet->biases[i], npl[i+1] * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    for (i = 0; i < training_set->num_examples; i++) 
    {
        // Copy the training data from host to device
        cudaErrorCheck(cudaMemcpy((void *)(d_training_inputs + i * INPUT_LAYER_SIZE), training_set->inputs[i], INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy((void *)(d_training_outputs + i * OUTPUT_LAYER_SIZE), training_set->outputs[i], OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    }

    cudaMemcpy(d_chain_rule_vector, chain_rule_vector, largest_layer_size*sizeof(float), cudaMemcpyHostToDevice);

    printf("done\n");

    for (epoch = epochs; epoch--;)
    {
        printf("\rtraining... epoch %d/%d", epochs-epoch, epochs);
        //printf("\rtraining... %d/%d cost: %f\n", epochs-epoch, epochs, nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
        for (batch = 0; batch < num_mini_batches; batch++)
        {
            for (nthExample = batch * examples_per_batch; nthExample < (batch + 1) * examples_per_batch; nthExample++)
            {
                nnet_backprop(d_weights, d_biases, d_activations, d_weight_gradients, d_bias_gradients, d_chain_rule_vector, (float *)(d_training_inputs + nthExample * INPUT_LAYER_SIZE), (float *)(d_training_outputs + nthExample * OUTPUT_LAYER_SIZE));
            }
            nnet_subtract_gradients(d_weights, d_weight_gradients, d_biases, d_bias_gradients, learn_rate, examples_per_batch);
        }
        for (; nthExample < training_set->num_examples; nthExample++)
        {
            nnet_backprop(d_weights, d_biases, d_activations, d_weight_gradients, d_bias_gradients, d_chain_rule_vector, (float *)(d_training_inputs + nthExample * INPUT_LAYER_SIZE), (float *)(d_training_outputs + nthExample * OUTPUT_LAYER_SIZE));
        }
        nnet_subtract_gradients(d_weights, d_weight_gradients, d_biases, d_bias_gradients, learn_rate, examples_per_batch);
    }
    printf("done\n");

    //copy weights and biases from device back to host's neural network struct
    for (int i = 0; i < LAYERS; i++)
    {
        for (int j = 0; j < npl[i+1]; j ++)
            cudaErrorCheck(cudaMemcpy(nnet->weights[i][j], (void *)(d_weights[i] + j * npl[i]), npl[i] * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(nnet->biases[i], d_biases[i], npl[i+1] * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // free host memory
    for (i = 0; i < LAYERS; i++)
    {
        laa_freeVector(weight_gradients[i]);
        laa_freeVector(bias_gradients[i]);
        laa_freeVector(activations[i]);
    }
    laa_freeVector(chain_rule_vector);

    // free device memory
    nnet_free_gpu_wba(d_weights, d_weight_gradients, d_biases, d_bias_gradients, d_activations);
    nnet_free_gpu_data(d_training_inputs, d_training_outputs);
    cudaFree(d_chain_rule_vector);
    return 1;
}

//=============================

void nnetcpu_layer_function_dense_relu(float **weights, int rows, int columns, float *activations, float *bias, float *destination)
{
    for (int i = 0; i < rows; i++)
        destination[i] = relu(laa_dot(activations, weights[i], columns) + bias[i]);
}

float *nnetcpu_feed_forward(float *inputs, NeuralNet *nnet, float *activations[LAYERS])
{
    int i;
    //hidden layers:
    nnetcpu_layer_function_dense_relu(nnet->weights[0], npl[1], npl[0], inputs, nnet->biases[0], activations[0]);
    for (i = 1; i < LAYERS-1; i++)
        nnetcpu_layer_function_dense_relu(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);

    //last/output layer:
    nnetcpu_layer_function_dense_relu(nnet->weights[i], npl[i + 1], npl[i], activations[i - 1], nnet->biases[i], activations[i]);

    return activations[LAYERS - 1];
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
        if (laa_maxIndexValue(test_set->outputs[i], npl[LAYERS]) != laa_maxIndexValue(nnetcpu_feed_forward(test_set->inputs[i], nnet, activations), npl[LAYERS]))
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

