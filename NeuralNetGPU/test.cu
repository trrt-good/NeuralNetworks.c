
__device__ void nnet_device_layer_function_dense_relu(float *d_weights, int rows, int columns, float *d_activations, float *d_bias, float *d_destination)
{
    float temp_sum = 0.0;

    for (int i = 0; i < rows; i++)
    {
        temp_sum = 0.0;
        for (int j = 0; j < columns; j ++)
        {
            temp_sum += d_weights[i * columns + j] * d_activations[j];
        }
        temp_sum += d_bias[i];
        d_destination[i] = temp_sum * (temp_sum > 0);
    }
}

__device__ float *nnet_device_feed_forward(float *inputs, float* d_weights[LAYERS], float *d_biases[LAYERS], float *activations[LAYERS])
{
    int i;
    //hidden layers:
    nnet_device_layer_function_dense_relu(d_weights[0], npl[1], npl[0], inputs, d_biases[0], activations[0]);
    for (i = 1; i < LAYERS-1; i++)
        nnet_device_layer_function_dense_relu(d_weights[i], npl[i + 1], npl[i], activations[i - 1], d_biases[i], activations[i]);

    //last/output layer:
    nnet_device_layer_function_dense_relu(d_weights[i], npl[i + 1], npl[i], activations[i - 1], d_biases[i], activations[i]);

    return activations[LAYERS - 1];
}

// calculates the gradient of the weights and biases of the current layer by multiplying their derivatives with respect to the current layer by the 
// current chain-rule product. 
__device__ void nnet_device_layer_function_dense_deriv_weights_and_biases(float *current_chain_deriv, int current_layer, float *activation, float **weight_gradients[LAYERS], float *bias_gradients[LAYERS])
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
__device__ void nnet_device_layer_function_dense_deriv_activations(NeuralNet* nnet, float* current_chain_deriv, float* math_buffer, int current_layer)
{
    multiply_MtVreplace(nnet->weights[current_layer], npl[current_layer+1], npl[current_layer], current_chain_deriv, math_buffer);
} 

__device__ void nnet_device_activation_function_deriv_relu(float* current_chain_deriv, int current_layer, float* activation)
{
    for (int i = 0; i < npl[current_layer+1]; i ++)
        current_chain_deriv[i] *= (activation[i] > 0);
}

__device__ void nnet_device_activation_function_deriv_sigmoid(float* current_chain_deriv, int current_layer, float* activation)
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
__device__ void nnet_device_cost_function_deriv_MSE(float* destination, float *activations, float* training_output)
{
    for (int i = 0; i < npl[LAYERS]; i ++)
        destination[i] = activations[i] - training_output[i];
}

__global__ void nnet_kernel_backprop(float *d_weights[LAYERS], float *d_biases[LAYERS], float *d_activations[LAYERS], float *d_weight_gradients[LAYERS], float *d_bias_gradients[LAYERS], float *d_current_chain_deriv, float **d_training_input, float **d_training_output, int batch_size)
{
    int layer = LAYERS-1; 
    float* training_input=d_training_input[threadIdx.x + blockDim.x * blockIdx.x];
    float* d_training_output=d_training_output[threadIdx.x + blockDim.x * blockIdx.x];

    nnet_device_feed_forward(training_input, d_weights, d_biases, d_activations);

    //set the current chain rule derivative value to the derivative of the cost function with respect to the last activation.
    nnet_cost_function_deriv_MSE(d_current_chain_deriv, d_activations[LAYERS-1], training_input);

    for (layer = LAYERS - 1; layer > 0; layer--)
    {
        //applies derivative of activation function (relu, sigmoid, etc) to the current_chain_deriv, in accordance to the chain rule
        nnet_device_activation_function_deriv_relu(d_current_chain_deriv, layer, d_activations[layer]);

        //updates the weight and bias gradients based on the current_chain_deriv.
        nnet_device_layer_function_dense_deriv_weights_and_biases(d_current_chain_deriv, layer, d_activations[layer-1], d_weight_gradients, d_bias_gradients);

        //updates the current_chain_deriv matrix for the next layer function derivative.
        nnet_device_layer_function_dense_deriv_activations(nnet, current_chain_deriv, math_buffer, layer);
    }
    //the first layer
    nnet_activation_function_deriv_relu(current_chain_deriv, layer, activations[layer]);
    nnet_layer_function_dense_deriv_weights_and_biases(current_chain_deriv, layer, training_input, weight_gradients, bias_gradients);
    nnet_layer_function_dense_deriv_activations(nnet, current_chain_deriv, math_buffer, layer);
}


int nnet_optimize2(NeuralNet *nnet, TrainingSet *training_set, int num_mini_batches, int epochs, float learn_rate)
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
    
    //TODO: check for copying errors
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
            nthExample = batch * examples_per_batch;
            nnet_kernel_backprop(d_weights, d_biases, d_activations, d_weight_gradients, d_bias_gradients, d_chain_rule_vector, (float *)(d_training_inputs + nthExample * INPUT_LAYER_SIZE), (float *)(d_training_outputs + nthExample * OUTPUT_LAYER_SIZE),examples_per_batch);
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
