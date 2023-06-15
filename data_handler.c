#include "data_handler.h"
#include "linear_alg.h"
#include "neural_net.h"

TestingSet *dh_testing_set_init(int num_testing_examples)
{
    TestingSet *new_set = malloc(sizeof(TestingSet));
    new_set->num_examples = num_testing_examples;
    int i;
    new_set->inputs = laa_allocMatrix(num_testing_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_testing_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
}

TrainingSet *dh_training_set_init(int num_training_examples)
{
    TrainingSet *new_set = malloc(sizeof(TrainingSet));
    new_set->num_examples = num_training_examples;
    int i;
    new_set->inputs = laa_allocMatrix(num_training_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_training_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
}

void dh_free_test_set(TestingSet *set)
{
    laa_freeMatrix(set->inputs, set->num_examples);
    laa_freeMatrix(set->outputs, set->num_examples);
    free(set);
}

void dh_free_training_set(TrainingSet *set)
{
    laa_freeMatrix(set->inputs, set->num_examples);
    laa_freeMatrix(set->outputs, set->num_examples);
    free(set);
}

void dh_shuffle_data(float** data_inputs, float **data_outputs, int num_data_points)
{
    srand(time(NULL));
    for (int i = num_data_points - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        float *tmp1 = data_inputs[i];
        data_inputs[i] = data_inputs[j];
        data_inputs[j] = tmp1;
        float *tmp2 = data_outputs[i];
        data_outputs[i] = data_outputs[j];
        data_outputs[j] = tmp2;
    }
}

void swap(float **arr, int i, int j)
{
    if (i == j)
        return;
    float *temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

/**
 * @brief reads data from the iris dataset (csv format, unaltered), and 
 * puts the result in the TrainingSet and TestingSet structs, which 
 * are assumed to be already initialized.
 * 
 * @param fileName name of the text file containing the data e.g. "iris.txt"
 * @param training_set the training data struct
 * @param testing_set the testing data struct
 * @return true if reading data was a success, false otherwise 
 */
int dh_read_data_iris(char *fileName, TrainingSet* training_set, TestingSet* testing_set)
{
    int total_data_points = training_set->num_examples + testing_set->num_examples;
    float **all_inputs = laa_allocMatrix(total_data_points, 4, 0);
    float **all_outputs = laa_allocMatrix(total_data_points, 3, 0);

    FILE *file = fopen(fileName, "r");
    if (file == NULL)
    {
        printf("Could not open file %s", fileName);
        return 0;
    }
    
    char line[1024];
    int index = 0;
    while (fgets(line, sizeof(line), file))
    {
        // Allocate memory for a new data point
        float* inputs = (float*)malloc(4 * sizeof(float));
        float* outputs = (float*)malloc(3 * sizeof(float));
        outputs[0] = 0.0f; outputs[1] = 0.0f; outputs[2] = 0.0f;

        // Parse the line
        char* token;

        token = strtok(line, ",");
        if (token == NULL || token[0] == '\n')
            break;
        for(int i = 0; i < 4; i++) {
            inputs[i] = atof(token);
            token = strtok(NULL, ",");
        }

        // one-hot-encoding
        if(strcmp(token, "Iris-setosa\n") == 0)
            outputs[0] = 1.0;
        else if(strcmp(token, "Iris-versicolor\n") == 0)
            outputs[1] = 1.0;
        else if(strcmp(token, "Iris-virginica\n") == 0)
            outputs[2] = 1.0;

        // Split the data into the training set and the testing set
        all_inputs[index] = inputs;
        all_outputs[index] = outputs;

        index++;
    }

    dh_shuffle_data(all_inputs, all_outputs, total_data_points);

    for (index = 0; index < total_data_points; index++)
    {
        if(index < training_set->num_examples)
        {
            training_set->inputs[index] = all_inputs[index];
            training_set->outputs[index] = all_outputs[index];
        }
        else
        {
            testing_set->inputs[index - training_set->num_examples] = all_inputs[index];
            testing_set->outputs[index - training_set->num_examples] = all_outputs[index];
        }
    }
    
    fclose(file);
    return 1;
}

int read_mnist_number_data(char *filename, int dataPoints, float **inputs, float **outputs)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }
    char line[4096];
    fgets(line, 4096, fp); // skip the first line
    fgets(line, 4096, fp); // skip the first line

    int i = 0;
    while (fgets(line, 4096, fp) && i < dataPoints)
    {
        char *tmp = strdup(line);

        // get the label
        int label = atoi(strtok(tmp, ","));
        outputs[i][label] = 1.0;

        // get the inputs
        for (int j = 0; j < 784; j++)
        {
            inputs[i][j] = atof(strtok(NULL, ","))/255.0;
        }

        free(tmp);
        i++;
    }

    fclose(fp);
}