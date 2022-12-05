#ifndef NEURAL_NET
#define NEURAL_NET

//use gradient descent to find the optimal learn rate
#define LEARN_RATE 0.007
#define ITERATIONS 5000

// The number of normal neural network layers, which does not include 
// the input layer or any convolution layers
#define LAYERS 3 

#define NUM_TRAINING_EXAMPLES 120     // number of data points used for training
#define NUM_TESTING_EXAMPLES 30       // number of data points used for testing

#define MINI_BATCHES 10
#define EXAMPLES_PER_BATCH (NUM_TRAINING_EXAMPLES/MINI_BATCHES)

void printNeuralNet();
void initNetwork();
void loadData(char* fileName, char* delimiter, int bufferSize);
int backProp(int showCost);
void resetNetwork();
float comparePredictions(int printEachTest, int printResults);
int saveToFile(const char* fileName);
int loadNetworkFromFile(const char* fileName);
#endif