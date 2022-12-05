#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PREPROCESS_TRAINING_DATA 1
#define N_FEATURES 4
#define N_TRAINING_EXAMPLES 4
#define LEARNING_RATE 1

//inv(X'*X)*X'*Y would do the same thing 

//first columns must be 1s because of dot product
float TRAINING_INPUTS[N_TRAINING_EXAMPLES][N_FEATURES+1] = 
{
    {1, 2104, 5, 1, 45},
    {1, 1416, 3, 2, 40},
    {1, 1534, 3, 2, 30},
    {1, 852, 2, 1, 36}
};

float TRAINING_OUTPUTS[N_TRAINING_EXAMPLES] = 
{
    460,
    232,
    315,
    178
};

//input to test the trained model
float TEST_INPUT[5] = {1, 2104, 5, 1, 45};

float dot(int vectorSizes, float a[], float b[])
{
    int index;
    float sum = 0;
    for (index = 0; index < vectorSizes; index++)
    {
        sum += a[index] * b[index];
    }
    return sum;
}

unsigned long getMicrotime()
{
	struct timeval currentTime;
	mingw_gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

void gradientDescent()
{
    int features = N_FEATURES + 1;
    //innitialize paramters and the temp paramter array which acts as a sort of buffer for training
    float parameters[features];
    float tempParameters[features];
    int i;
    int j;

#if PREPROCESS_TRAINING_DATA
    float featureScaling[features];
    float meanNormalization[features];
    float avg;
    float max;
    float min;
#endif

    
    for (i = 0; i < features; i++)
    {
        parameters[i] = 0;
        tempParameters[i] = 0;
#if PREPROCESS_TRAINING_DATA
        avg = 0;
        max = -1000000000000;
        min = 1000000000000;
        for (j = 0; j < N_TRAINING_EXAMPLES; j++)
        {
            avg += TRAINING_INPUTS[j][i]/N_TRAINING_EXAMPLES;

            if (TRAINING_INPUTS[j][i] > max)
                max = TRAINING_INPUTS[j][i];
            else if (TRAINING_INPUTS[j][i] < min)
                min = TRAINING_INPUTS[j][i];
        }
        featureScaling[i] = (min-max)/2;
        meanNormalization[i] = -avg;
        for (j = 0; j < N_TRAINING_EXAMPLES; j++)
        {
            TRAINING_INPUTS[j][i] = (TRAINING_INPUTS[j][i] + meanNormalization[i])/featureScaling[i];
        }
#endif
    }

#if PREPROCESS_TRAINING_DATA
    for (j = 0; j < N_TRAINING_EXAMPLES; j++)
    {
        TRAINING_INPUTS[j][0] = 1;
    }
#endif 

    for (i = 0; i < N_TRAINING_EXAMPLES; i ++)
    {
        for (j = 0; j < features; j ++)
        {
            printf("%f, ", TRAINING_INPUTS[i][j]);
        }
        printf("\n");
    }

    int iterate;
    float singleParamGradient;
    int nthTrainingExample;
    int nthParam;

    unsigned long start;
    start = getMicrotime();

    for (iterate = 10000; iterate--; )
    {
        for (nthParam = 0; nthParam < features; nthParam++)
        {
            for (nthTrainingExample = 0; nthTrainingExample < N_TRAINING_EXAMPLES; nthTrainingExample++)
            {
                singleParamGradient += (dot(features, parameters, TRAINING_INPUTS[nthTrainingExample]) - TRAINING_OUTPUTS[nthTrainingExample])*TRAINING_INPUTS[nthTrainingExample][nthParam];
            }
            singleParamGradient /= N_TRAINING_EXAMPLES;
            tempParameters[nthParam] = tempParameters[nthParam] - LEARNING_RATE*singleParamGradient;
        }
        for (i = 0; i < features; i++)
        {
            parameters[i] = tempParameters[i];
        }
    }

    double dt;
    dt = (double)(getMicrotime()-start)/1000000;
    printf("\ntrainingTime: %.10f seconds\n", dt);

#if PREPROCESS_TRAINING_DATA
    float tempArr[N_FEATURES+1];
    tempArr[0] = 1;
    for (i = 1; i < features; i++)
    {
        tempArr[i] = (TEST_INPUT[i] + meanNormalization[i])/featureScaling[i];
    }
#endif

    printf("\ntrained function parameters: ");
    for (i = 0; i < features; i ++)
    {
        printf("%f, ", parameters[i]);
    }

    printf("\n\n");

#if PREPROCESS_TRAINING_DATA
    printf("\nvalue: %f\n\n", dot(features, parameters, tempArr));
#else 
    printf("\nvalue: %f\n\n", dot(features, parameters, TEST_INPUT));
#endif
};

int main()
{
    gradientDescent();

    float tempV[5] = {
        188.4,
        0.38663,
        -56.138,
        -92.967,
        -3.7378
    };

    float tempV1[5] = {1, 2104, 5, 1, 45};
    
    printf("\n%f\n", dot(5, tempV, tempV1));

    return 0;
};

//TODO: read data from file, stocastic gradient descent, make algorithm that creates a bunch of data. 
