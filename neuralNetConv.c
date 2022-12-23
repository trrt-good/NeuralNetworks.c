#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <windows.h>
#include <profileapi.h>
#include "..\\LinearAlg\\linearAlg.h"
#include "..\\FileIO\\dataReader.h"
#include "neuralNetConv.h"

#define RELU(a) ((a<0)? 0 : 1)

/**
 * @brief npl = NODES PER LAYER. this is the central array
 * which the neural network is built from.
 * Index 0 of the array is how many input neurons there are.
 * The last index is how many output neurons there are.
 * The numbers in npl from index 1 to CONV_LAYERS represent the width
 * of the convolution layer at that index, assuming the convolution
 * layer is a square. Does not support non square layers. 
 */
const int npl[LAYERS + 1] = {100, 25, 15, 10};
const int npc[CONV_LAYERS + 1] = {28, 12, 10};
const int spk[CONV_LAYERS] = {5, 3}; //width of each square kernel. Does not support non square kernels
//the stride of each kernel. The horizontal and vertical stride must be the same, which is why there is only one number for each kernel. 
const int strides[CONV_LAYERS] = {2, 1}; 

float *trainingInputs[NUM_TRAINING_EXAMPLES];
float *trainingOutputs[NUM_TRAINING_EXAMPLES];

float *testingInputs[NUM_TESTING_EXAMPLES];
float *testingOutputs[NUM_TESTING_EXAMPLES];

// even though kernels are usually represented as a 2d matrix, because the 
// convolution operation is basically a dot product, it isn't necessary to 
// use a 2d array.
float *kernels[CONV_LAYERS]; 
float **kernelGradients[CONV_LAYERS]; // the acutal gradient is only stored in the first row of each 2d matrix. A matrix is used for calculating the gradient.

float **weights[LAYERS];
float **weightGradients[LAYERS]; // holds the changes to weights per iteration

float *biases[LAYERS];
float *biasGradients[LAYERS]; // holds the changes to biases per iteration

float *activations[LAYERS];
float *activationsConv[CONV_LAYERS]; //even though these can be thought of as 2d images, it's easier to code as a vector. 

//function declairations
void computeOneActivation(float **matrix, int rows, int columns, float *multVect, float *addVect, float *destination);
void multiplyReplace(float **a_matrixVals, int a_rows, int a_columns, float **b_matrixVals, int b_rows, int b_columns, float* buffer);
void computeLast2LayerGradients(float *outputData);
void subtractGradients();
void runData(float inputs[npl[0]]);

void verifyNetwork()
{
    int i;
    if (npl[0] != npc[CONV_LAYERS]*npc[CONV_LAYERS])
    {
        printf("npl[0] must equal npc[CONV_LAYERS]*npc[CONV_LAYERS]");
        exit(1);
    }
    for (i = 0; i < CONV_LAYERS; i ++)
    {
        if (!(spk[i]&1)) //if the kernel size is even
        {
            printf("kernel size must be odd!");
            exit(1);
        }
        int correctSize = (npl[i] - (spk[i]-1)) / (strides[i]*strides[i]);
        if (npc[i+1] != correctSize) //if the size of the convolution layer is wrong
        {
            printf("Invalid convolution layer size! based on the kernel and stride, the convolution layer at npl[%d] should be %d", i+1, correctSize);
            exit(1);
        }
    }
}

/**
 * @brief initializes the network's basic layers, convolution layers, kernels, 
 * weights biases and the gradients for each parameter.
 * 
 */
void initNetwork()
{
    srand(time(NULL));
    int i;
    for (i = 0; i < NUM_TRAINING_EXAMPLES; i++)
    {
        trainingInputs[i] = laa_allocVectorRaw(npc[0]*npc[0]);
        trainingOutputs[i] = laa_allocVectorRaw(npl[LAYERS]);
    }
    for (i = 0; i < NUM_TESTING_EXAMPLES; i++)
    {
        testingInputs[i] = laa_allocVectorRaw(npc[0]*npc[0]);
        testingOutputs[i] = laa_allocVectorRaw(npl[LAYERS]);
    }

    for (i = 0; i < CONV_LAYERS; i ++)
    {
        kernels[i] = laa_allocRandVector(spk[i]*spk[i]);
        kernelGradients[i] = laa_allocVector(spk[i]*spk[i], 0);

        activationsConv[i] = laa_allocVector(npc[i+1]*npc[i+1], 0);
    }

    for (i = 0; i < LAYERS; i++)
    {
        weights[i] = laa_allocRandMatrix(npl[i + 1], npl[i]);
        weightGradients[i] = laa_allocMatrix(npl[i + 1], npl[i], 0);

        biases[i] = laa_allocRandVector(npl[i + 1]);
        biasGradients[i] = laa_allocVector(npl[i + 1], 0);

        activations[i] = laa_allocVector(npl[i + 1], 0);
    }
}

void resetNetwork()
{
    srand(rand()*time(NULL)%0xffffffff);
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        laa_setMatrixToRand(weights[i], npl[i + 1], npl[i]);
        laa_setMatrixTo(weightGradients[i], npl[i + 1], npl[i], 0);

        laa_setVectorToRand(biases[i], npl[i + 1]);
        laa_setVectorTo(biasGradients[i], npl[i + 1], 0);

        laa_setVectorTo(activations[i], npl[i + 1], 0);
    }
}

void loadData(char* fileName, char* delimiter, int bufferSize)
{
    readRowData_ML(fileName, delimiter, bufferSize, NUM_TRAINING_EXAMPLES + NUM_TESTING_EXAMPLES, NUM_TESTING_EXAMPLES, npl[0], npl[LAYERS], trainingInputs, trainingOutputs, testingInputs, testingOutputs, 1);
    // laa_printMatrix(trainingInputs, NUM_TRAINING_EXAMPLES, npl[0]);
    // laa_printMatrix(trainingOutputs, NUM_TRAINING_EXAMPLES, npl[LAYERS]);
    // laa_printMatrix(testingInputs, NUM_TESTING_EXAMPLES, npl[0]);
    // laa_printMatrix(testingOutputs, NUM_TESTING_EXAMPLES, npl[LAYERS]);
}

void runData(float inputs[npc[0]])
{
    computeConvolutionLayer(inputs, npc[0], kernels[0], spk[0], strides[0], activationsConv[0])
    for (int i = 1; i < CONV_LAYERS; i ++)
    {
        computeConvolutionLayer(activationsConv[i-1], npc[i], kernels[i], spk[i], strides[i], activationsConv[i])
    }
    computeOneActivation(weights[0], npl[1], npl[0], activationsConv[CONV_LAYERS-1], biases[0], activations[0]);
    int i;
    for (i = 1; i < LAYERS; i++)
    {
        computeOneActivation(weights[i], npl[i + 1], npl[i], activations[i - 1], biases[i], activations[i]);
    }
}

float cost(float activation[npl[LAYERS]], float y[npl[LAYERS]])
{
    float sum = 0;
    int i;
    for (i = 0; i < npl[LAYERS]; i ++)
    { 
        sum += (y[i] - activation[i])*(y[i] - activation[i]);
    }
    return sum;
}

float totalCost()
{
    float sum = 0;
    int i, j;
    for (i = 0; i < NUM_TRAINING_EXAMPLES; i ++)
    {
        runData(trainingInputs[i]);
        for (j = 0; j < npl[LAYERS]; j ++)
        { 
            sum += (trainingOutputs[i][j] - activations[LAYERS-1][j])*(trainingOutputs[i][j] - activations[LAYERS-1][j])/NUM_TRAINING_EXAMPLES;
        }
    }
    return sum;
}
int backProp(int showCost)
{
    // const int EXAMPLES_PER_BATCH = (MINI_BATCHES)? NUM_TRAINING_EXAMPLES/MINI_BATCHES : 0;
    int largestLayerSize = 0;
    int iteration;
    int nthExample;
    int layer;
    int batch;
    int i, j, k, l;
    float dotSum1 = 0;
    float dotSum2 = 0;
    float diff;
    for (i = LAYERS+1; i--;)
        largestLayerSize = max(largestLayerSize, npl[i]);
    float weightProductBuffer[largestLayerSize];
    //weightProduct holds the cumulative product of the weights temporarily during the calculation of the weight and bias gradients. 
    float **weightProduct = laa_allocMatrix(largestLayerSize, largestLayerSize, 0);
    
    for (iteration = ITERATIONS; iteration--;)
    {
        if (showCost)
            printf("\rcost: %f", totalCost());
        for (batch = 0; batch < MINI_BATCHES; batch++)
        {
            for (nthExample = batch*EXAMPLES_PER_BATCH; nthExample < (batch+1)*EXAMPLES_PER_BATCH; nthExample++)
            {
                runData(trainingInputs[nthExample]);

                // compute the last layer two gradients so that the rest can be calculated in a simple algorithm 
                computeLast2LayerGradients(trainingOutputs[nthExample]);

                // set the weight product to the last layer's weight
                laa_copyMatrixValues(weights[LAYERS-1], weightProduct, npl[LAYERS], npl[LAYERS-1]);

                for (layer = LAYERS-2; layer > 1; layer--)
                {
                    multiplyReplace(weightProduct, npl[layer+2], npl[layer+1], weights[layer], npl[layer+1], npl[layer], weightProductBuffer);
                    for (i = 0; i < npl[layer]; i ++)
                    {
                        for (j = 0; j < npl[layer-1]; j++)
                        {
                            dotSum1 = 0;
                            for (k = 0; k < npl[LAYERS]; k++)
                            {
                                dotSum1 += weightProduct[k][i]*activations[layer-1][j]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                            }
                            weightGradients[layer-1][i][j] += LEARN_RATE*dotSum1;
                        }

                        dotSum2 = 0;
                        for (k = 0; k < npl[LAYERS]; k++)
                        {
                            dotSum2 += weightProduct[k][i]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                        }
                        biasGradients[layer-1][i] += LEARN_RATE*dotSum2;
                    }
                }
                multiplyReplace(weightProduct, npl[layer+2], npl[layer+1], weights[layer], npl[layer+1], npl[layer], weightProductBuffer);
                for (i = 0; i < npl[layer]; i ++)
                {
                    for (j = 0; j < npl[layer-1]; j++)
                    {
                        dotSum1 = 0;
                        for (k = 0; k < npl[LAYERS]; k++)
                        {
                            dotSum1 += weightProduct[k][i]*activationsConv[CONV_LAYERS-1][j]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                        }
                        weightGradients[layer-1][i][j] += LEARN_RATE*dotSum1;
                    }

                    dotSum2 = 0;
                    for (k = 0; k < npl[LAYERS]; k++)
                    {
                        dotSum2 += weightProduct[k][i]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                    }
                    biasGradients[layer-1][i] += LEARN_RATE*dotSum2;
                }
                //gradient calculation for convolution layers
                for (layer = CONV_LAYERS-1; layer >= 0; layer --)
                {
                    for (i = 0; i < npc[layer+1]; i ++)
                    {
                        for (j = 0; j < npc[layer+1]; j ++)
                        {
                            for (k = 0; k < spk[layer]*spk[layer]; k ++)
                            {
                                dotSum1 += activationsConv[layer][(i*npc[layer] + j) + k%spk[layer] + k/spk[layer]*npc[layer]]*weightProduct[]; //loops through the weights 
                            }
                        }
                    }
                }
            }
            subtractGradients();
        }
        for (nthExample = nthExample; nthExample < NUM_TRAINING_EXAMPLES; nthExample++)
        {
            runData(trainingInputs[nthExample]);

            // compute the last layer two gradients so that the rest can be calculated in a simple algorithm 
            computeLast2LayerGradients(trainingOutputs[nthExample]);

            // set the weight product to the last layer's weight
            laa_copyMatrixValues(weights[LAYERS-1], weightProduct, npl[LAYERS], npl[LAYERS-1]);

            for (layer = LAYERS-2; layer > 1; layer--)
            {
                multiplyReplace(weightProduct, npl[layer+2], npl[layer+1], weights[layer], npl[layer+1], npl[layer], weightProductBuffer);
                for (i = 0; i < npl[layer]; i ++)
                {
                    for (j = 0; j < npl[layer-1]; j++)
                    {
                        dotSum1 = 0;
                        for (k = 0; k < npl[LAYERS]; k++)
                        {
                            dotSum1 += weightProduct[k][i]*activations[layer-1][j]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                        }
                        weightGradients[layer-1][i][j] += LEARN_RATE*dotSum1;
                    }

                    dotSum2 = 0;
                    for (k = 0; k < npl[LAYERS]; k++)
                    {
                        dotSum2 += weightProduct[k][i]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                    }
                    biasGradients[layer-1][i] += LEARN_RATE*dotSum2;
                }
            }
            multiplyReplace(weightProduct, npl[layer+2], npl[layer+1], weights[layer], npl[layer+1], npl[layer], weightProductBuffer);
            for (i = 0; i < npl[layer]; i ++)
            {
                for (j = 0; j < npl[layer-1]; j++)
                {
                    dotSum1 = 0;
                    for (k = 0; k < npl[LAYERS]; k++)
                    {
                        dotSum1 += weightProduct[k][i]*trainingInputs[nthExample][j]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                    }
                    weightGradients[layer-1][i][j] += LEARN_RATE*dotSum1;
                }

                dotSum2 = 0;
                for (k = 0; k < npl[LAYERS]; k++)
                {
                    dotSum2 += weightProduct[k][i]*(activations[LAYERS-1][k]-trainingOutputs[nthExample][k]);
                }
                biasGradients[layer-1][i] += LEARN_RATE*dotSum2;
            }
        }
        subtractGradients();

        if (isnan(activations[0][0]))
        {
            return 0;
        } 
        
        //subtract gradients from the weights and biases

    }
    return 1;
}

void computeLast2LayerGradients(float *outputData)
{
    int i, j, k;
    float diff, sum;
    for (i = 0; i < npl[LAYERS]; i++)
    {
        diff = activations[LAYERS - 1][i] - outputData[i];
        biasGradients[LAYERS - 1][i] += LEARN_RATE * diff;
        for (j = 0; j < npl[LAYERS - 1]; j++)
        { 
            weightGradients[LAYERS - 1][i][j] += LEARN_RATE * diff * activations[LAYERS - 2][j];

            //because it's += this essentially acts like a dot product for each row in the 
            // bias gradient, completing the necessary vector-matrix multiplcation while still using
            // the same loops.
            biasGradients[LAYERS-2][j] += LEARN_RATE*weights[LAYERS-1][i][j]*diff;

            //dot product:
            for (k = 0; k < npl[LAYERS-2]; k++)
            {
                weightGradients[LAYERS-2][j][k] += LEARN_RATE * weights[LAYERS-1][i][j] * diff * activations[LAYERS - 3][k];
            }
        }
    }
}

void subtractGradients()
{
    //(RELU(activations[layer][i]))
    int layer, i, j; 
    for (layer = 0; layer < LAYERS; layer++)
    {
        for (i = 0; i < npl[layer + 1]; i++)
        {
            for (j = 0; j < npl[layer]; j++)
            {
                weights[layer][i][j] -= weightGradients[layer][i][j]/ NUM_TRAINING_EXAMPLES;
                weightGradients[layer][i][j] = 0;
            }
            biases[layer][i] -= biasGradients[layer][i] / NUM_TRAINING_EXAMPLES;
            biasGradients[layer][i] = 0;
        }
    }
}

int largestIndex(float* arr, int n)
{
    int i, index = 0;
    float max = arr[0];
    
    for (i = 0; i < n; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
            index = i;
        }
    }
    return index;
}

float comparePredictions(int printEachTest, int printResults)
{
    int i = 0, numWrong = 0;
    const unsigned int testNum = (NUM_TESTING_EXAMPLES)? NUM_TESTING_EXAMPLES : NUM_TRAINING_EXAMPLES;
    for (i = 0; i < testNum; i ++)
    {
        if (NUM_TESTING_EXAMPLES)
        {
            runData(testingInputs[i]);
            if (largestIndex(testingOutputs[i], npl[LAYERS]) != largestIndex(activations[LAYERS-1], npl[LAYERS]))
                numWrong++;
        }
        else
        {
            runData(trainingInputs[i]);
            if (largestIndex(trainingOutputs[i], npl[LAYERS]) != largestIndex(activations[LAYERS-1], npl[LAYERS]))
                numWrong++;
        }
            
        if (printEachTest && printResults)
        {
            printf("\ndesired output: ");
            laa_printVector((testingOutputs == NULL)? trainingOutputs[i] : testingOutputs[i], npl[LAYERS]);
            printf("predicted output: ");
            laa_printVector(activations[LAYERS-1], npl[LAYERS]);
        }
    }
    if (printResults)
    {
        printf("total cost: %f\n", totalCost());
        printf("accuracy: %d/%d (%.3f%%)\n", testNum-numWrong, testNum, 100*(testNum-numWrong)/(float)testNum);
    }
    
    return (testNum-numWrong)/(float)testNum;
}

// --- --- --- --- --- --- --- --- File IO  --- --- --- --- --- --- --- --- 
#pragma region FileIO

void printNeuralNet()
{
    int i;
    for (i = 0; i < LAYERS; i++)
    {
        printf("\nweights:");
        laa_printMatrix(weights[i], npl[i + 1], npl[i]);
        printf("bias: ");
        laa_printVector(biases[i], npl[i + 1]);
    }
}

int saveToFile(const char* fileName)
{
    FILE* filePointer = fopen(fileName, "wb");
    if (filePointer == NULL)
        return 0;

    int layers = LAYERS, i;
    fwrite(&layers, sizeof(int), 1, filePointer);
    fwrite(npl, sizeof(int), layers+1, filePointer); 
    for (i = 0; i < LAYERS; i++)
    {
        laa_writeMatrixBin(weights[i], npl[i + 1], npl[i], filePointer);
        laa_writeVectorBin(biases[i], npl[i + 1], filePointer);
    }
    fclose(filePointer);
    return 1;   
}

int loadNetworkFromFile(const char* fileName)
{
    FILE* filePointer = fopen(fileName, "rb");
    if (filePointer == NULL)
    {
        return 0;
    }

    int layers, i;
    fread(&layers, sizeof(layers), 1, filePointer);
    int* npl = malloc(sizeof(int)*(layers+1));
    fread(npl, sizeof(int), layers+1, filePointer);

    float* biasVector = laa_allocVectorRaw(npl[i+1]);
    for (i = 0; i < layers; i++)
    {
        laa_readMatrixBin(weights[i], filePointer);
        laa_readVectorBin(biases[i], filePointer);
    }
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

void computeConvolutionLayer(float* input, int width, float* kernel, const int kernelSize, int stride, float* destination)
{
    int outputWidth = width-(kernelSize-1);
    int i, j, k;
    float dotSum = 0;
    for (i = 0; i < outputWidth; i ++)
    {
        for (j = 0; j < outputWidth; j ++)
        {
            for (k = 0; k < kernelSize*kernelSize; k ++)
            {
                dotSum += input[(i*width + j) + k%kernelSize + k/kernelSize*width]*kernel[k];
            }
            destination[i*outputWidth+j] = dotSum;
        }
    }
}

void multiplyReplace(float **a_matrixVals, int a_rows, int a_columns, float **b_matrixVals, int b_rows, int b_columns, float* buffer)
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