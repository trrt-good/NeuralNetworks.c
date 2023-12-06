# NeuralNetworks.c

This repository contains a simple machine learning package that includes implementations for both CPU and GPU. The package is designed to facilitate neural network operations leveraging the power of multicore CPUs and CUDA-enabled GPUs.

## Features

- **Cross-platform compatibility:** Runs on any platform that supports C/C++ and OpenMP/CUDA.
- **Dual support:** Offers both CPU and GPU versions for flexible application.
- **Ease of use:** Straightforward API for creating and managing neural networks.
- **Parallel processing:** Utilizes OpenMP in the CPU version for efficient computation.
- **GPU acceleration:** Harnesses the power of NVIDIA's CUDA for the GPU version.
- **Customizability:** Adjustable learning rates, epochs, and batch sizes.
- **Sample datasets:** Includes example datasets for training and testing.

## File Structure

The project is divided into two main directories:

- `NeuralNetCPU` - Contains the CPU-optimized version of the neural network.
- `NeuralNetGPU` - Contains the GPU-accelerated version using CUDA.

### NeuralNetCPU Contents:

- `bin/` - Compiled binaries and executables.
- `Data/` - Sample training data.
- `.vscode/` - VSCode configuration for building and debugging.

### NeuralNetGPU Contents:

- `bin/` - Compiled binaries and executables for the GPU version.
- `Data/` - Sample training data.
- `.vscode/` - VSCode configuration for building and debugging the GPU version.

## Requirements

For the CPU version:
- A C compiler with OpenMP support (e.g., GCC, Clang).
- OpenMP library for parallel processing.

For the GPU version:
- NVIDIA CUDA Toolkit for compiling and running CUDA code.

## Installation

Clone the repository:

```sh
git clone https://github.com/trrt-good/MachineLearning/tree/master/
```

Navigate to the cloned directory:

```sh
cd MachineLearning
```

## Usage

### CPU Version

The CPU version of the machine learning library uses OpenMP for parallel processing. Simply include `neural_net.h` in your C code and compile it with `-fopenmp`.

**Example Usage:**

```c
// main.c snippet
#include "stdio.h"
#include "neural_net.h"

#define LEARN_RATE 0.4
#define EPOCHS 10
 
#define NUM_TRAINING_EXAMPLES 60000     // number of data points used for training
#define NUM_TESTING_EXAMPLES 10000       // number of data points used for testing
 
#define BATCHES 150

// bounds for randomly initialized weights
#define INIT_MIN -0.1
#define INIT_MAX 0.1

int main()
{
    printf("Available Threads: %d\nThreads used: %d\n", omp_get_max_threads(), MAX_THREADS);

    NeuralNet* nnet = nnet_init(INIT_MIN, INIT_MAX);

    printf("loading data... ");
    // initialize data containers
    TrainingSet* training_set = dh_training_set_init(NUM_TRAINING_EXAMPLES);
    TestingSet* test_set = dh_testing_set_init(NUM_TESTING_EXAMPLES);

    // read mnist data
    dh_read_mnist_digits_images("Data/train-images.idx3-ubyte", training_set->num_examples, training_set->inputs);
    dh_read_mnist_digits_labels("Data/train-labels.idx1-ubyte", training_set->num_examples, training_set->outputs);
    dh_read_mnist_digits_images("Data/t10k-images.idx3-ubyte", test_set->num_examples, test_set->inputs);
    dh_read_mnist_digits_labels("Data/t10k-labels.idx1-ubyte", test_set->num_examples, test_set->outputs);
    printf("done\n");

    // train neural net
    nnet_optimize_parallel(nnet, training_set, BATCHES, EPOCHS, LEARN_RATE);

    // optional non parallel version
    // nnet_optimize(nnet, training_set, BATCHES, EPOCHS, LEARN_RATE);

    // evaluate
    printf("final cost: %f\n", nnet_total_cost(nnet, training_set->inputs, training_set->outputs, training_set->num_examples));
    nnet_test_results(nnet, test_set, 0, 1);

    // save to a file
    nnet_save_to_file(nnet, "bin/testNet.nnet");

    // load from the file for testing
    nnet_load_from_file(nnet, "bin/testNet.nnet");
    nnet_test_results(nnet, test_set, 0, 1);
}
```

**Compilation:**

```sh
gcc -fopenmp neural_net.c linear_alg.c data_handler.c main.c -o neural_net_cpu
```


### GPU Version

The GPU version uses CUDA for parallel computation on NVIDIA GPUs.

**Example Usage: (Basically same as CPU)**

```c
// main.c snippet
#include "stdio.h"
#include "neural_net_gpu.h"

#define LEARN_RATE 0.4
#define EPOCHS 10
 
#define NUM_TRAINING_EXAMPLES 60000     // number of data points used for training
#define NUM_TESTING_EXAMPLES 10000       // number of data points used for testing
 
#define BATCHES 150

// bounds for randomly initialized weights
#define INIT_MIN -0.1
#define INIT_MAX 0.1

int main()
{
    NeuralNet* nnet = nnet_init(INIT_MIN, INIT_MAX);

    printf("loading data... ");
    // initialize data containers
    TrainingSet* training_set = dh_training_set_init(NUM_TRAINING_EXAMPLES);
    TestingSet* test_set = dh_testing_set_init(NUM_TESTING_EXAMPLES);

    // read mnist data
    dh_read_mnist_digits_images("Data/train-images.idx3-ubyte", training_set->num_examples, training_set->inputs);
    dh_read_mnist_digits_labels("Data/train-labels.idx1-ubyte", training_set->num_examples, training_set->outputs);
    dh_read_mnist_digits_images("Data/t10k-images.idx3-ubyte", test_set->num_examples, test_set->inputs);
    dh_read_mnist_digits_labels("Data/t10k-labels.idx1-ubyte", test_set->num_examples, test_set->outputs);
    printf("done\n");

    // begin training
    nnet_optimize(nnet, training_set, BATCHES, EPOCHS, LEARN_RATE);
    
    nnet_test_results(nnet, testing_set, 0, 1);
    nnet_save_to_file(nnet, "bin/testNet.nnet");
}
```

**Compilation:**

```sh
nvcc neural_net_gpu.cu linear_alg.cu data_handler.cu main.cu -o neural_net_gpu
```

---

# API Reference

## `data_handler.h`

### Overview
The `data_handler.h` header file defines the structures and functions necessary for handling training and testing datasets in neural network applications. It includes functionalities for initializing datasets, reading data from files, and managing the memory associated with these datasets.

### Data Structures
1. **TrainingSet**: A structure that holds the training data including the number of examples, inputs, and corresponding outputs.
2. **TestingSet**: Similar to TrainingSet, this structure holds the testing data.

### Functions
- **Initialization and Memory Management**
    - `TrainingSet *dh_training_set_init(int num_training_examples)`: Initializes a training set with a specified number of examples.
    - `TestingSet *dh_testing_set_init(int num_testing_examples)`: Initializes a testing set with a specified number of examples.
    - `void dh_free_training_set(TrainingSet *set)`: Frees the memory allocated for a training set.
    - `void dh_free_testing_set(TestingSet *set)`: Frees the memory allocated for a testing set.

- **Data Handling**
    - `void dh_shuffle_data(float** data_inputs, float **data_outputs, int num_data_points)`: Shuffles the data points in a dataset.
    - `int dh_read_data_iris(const char *filename, TrainingSet* training_set, TestingSet* testing_set)`: Reads Iris dataset from a file and populates the training and testing sets.
    - `int dh_read_mnist_digits_images(const char *filename, int num_data_points, float **data)`: Reads MNIST digit images from a file.
    - `int dh_read_mnist_digits_labels(const char *filename, int num_data_points, float **data)`: Reads MNIST digit labels from a file.
    - `void dh_print_image(float* pixels, int image_width)`: Prints an image given its pixel data and width.
    - `int read_mnist_number_data(const char *filename, int dataPoints, float **inputs, float **outputs)`: Reads MNIST number data from a file.

## `neural_net.h`

### Overview
The `neural_net.h` header file contains the core structures and functions for defining and operating a neural network in C. It includes neural network initialization, training, testing, and utility functions.

### Constants
- Neural Network Architecture:
    - `LAYERS`: The number of layers in the neural network, excluding the input layer.
    - `INPUT_LAYER_SIZE`, `HIDDEN_LAYER_SIZES`, `OUTPUT_LAYER_SIZE`: Size configurations for the input, hidden, and output layers.
    - `MAX_THREADS`: Maximum number of threads for parallel operations.

### Data Structures
- **NeuralNet**: Represents the neural network with weights, biases, and layer information.

### Functions
- **Initialization and Memory Management**
    - `NeuralNet* nnet_init(float init_min, float init_max)`: Initializes a neural network with random weights and biases.
    - `void nnet_free(NeuralNet *nnet)`: Frees the memory allocated for a neural network.

- **Training and Testing**
    - `float* nnet_feed_forward(float *inputs, NeuralNet* nnet, float* activations[LAYERS])`: Performs feed-forward operation.
    - `float nnet_total_cost(NeuralNet* nnet, float** inputs, float** outputs, int n)`: Computes the total cost for a given set of inputs and outputs.
    - `int nnet_optimize(NeuralNet* nnet, TrainingSet* training_set, int num_mini_batches, int iterations, float learn_rate)`: Optimizes the neural network using backpropagation.
    - `float nnet_test_results(NeuralNet* nnet, TestingSet* test_set, int print_each_test, int print_results)`: Tests the neural network on a given testing set.

- **Utility Functions**
    - `void nnet_print(NeuralNet* nnet)`: Prints the neural network's structure and parameters.
    - `int nnet_save_to_file(NeuralNet* nnet, const char* fileName)`: Saves the neural network to a file.
    - `int nnet_load_from_file(NeuralNet* nnet, const char* fileName)`: Loads a neural network from a file.

## `neural_net_gpu.h`

### Overview
The `neural_net_gpu.h` header file extends the neural network functionalities to leverage GPU acceleration using CUDA. It includes specialized functions for allocating and managing GPU memory, as well as optimized training and testing routines.

### Constants
- GPU-related Configurations:
    - `MAX_LAYER_SIZE`, `BLOCK_SIZE`: Parameters for GPU memory allocation and kernel

 execution.

### Functions
- **GPU Memory Management**
    - `void nnet_alloc_gpu_wba(...)`: Allocates GPU memory for weights, biases, and activations.
    - `void nnet_free_gpu_wba(...)`: Frees the allocated GPU memory.

- **Training and Testing with GPU**
    - `void nnet_feed_forward(float *d_inputs, ...)`: GPU-accelerated feed-forward operation.
    - `float nnet_total_cost(float **correct_outputs, float ** predictions, int num_data_points)`: Computes the total cost for predictions.
    - `int nnet_optimize(NeuralNet* nnet, TrainingSet* training_set, int num_mini_batches, int iterations, float learn_rate)`: Optimizes the network using GPU-accelerated routines.

- **Utility Functions**
    - `void nnet_print(NeuralNet* nnet)`: Prints the neural network's structure and parameters.
    - `int nnet_save_to_file(NeuralNet* nnet, const char* fileName)`: Saves the neural network to a file.
    - `int nnet_load_from_file(NeuralNet* nnet, const char* fileName)`: Loads a neural network from a file.
