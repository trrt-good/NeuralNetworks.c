#include "data_handler.h"
#include "linear_alg.h"
#include "neural_net_gpu.h"

TestingSet *dh_testing_set_init(int num_testing_examples)
{
    TestingSet *new_set = (TestingSet *)malloc(sizeof(TestingSet));
    new_set->num_examples = num_testing_examples;
    new_set->inputs = laa_allocMatrix(num_testing_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_testing_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
}

TrainingSet *dh_training_set_init(int num_training_examples)
{
    TrainingSet *new_set = (TrainingSet *)malloc(sizeof(TrainingSet));
    new_set->num_examples = num_training_examples;
    new_set->inputs = laa_allocMatrix(num_training_examples, INPUT_LAYER_SIZE, 0);
    new_set->outputs = laa_allocMatrix(num_training_examples, OUTPUT_LAYER_SIZE, 0);
    return new_set;
}

void dh_free_testing_set(TestingSet *set)
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
int dh_read_data_iris(const char *filename, TrainingSet* training_set, TestingSet* testing_set)
{
    int total_data_points = training_set->num_examples + testing_set->num_examples;
    float **all_inputs = laa_allocMatrix(total_data_points, 4, 0);
    float **all_outputs = laa_allocMatrix(total_data_points, 3, 0);

    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Could not open file %s", filename);
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

void dh_print_image(float* pixels, int image_width)
{
    char shades[] = " .::#$&&%%%%%%";
    int num_shades = sizeof(shades) - 1;

    for (int i = 0; i < image_width; i++) 
    {
        for (int j = 0; j < image_width; j++) 
        {
            float pixel = pixels[i*28 + j];
            int shade_index = (int) (pixel * num_shades);

            // Print the shade twice
            printf("%c%c", shades[shade_index], shades[shade_index]);
        }
        printf("\n");
    }
}

// normalizes
int dh_read_mnist_digits_images(const char *filename, int num_data_points, float **data) 
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        return 0; // File not found
    }

    // Skip the header
    fseek(file, 16, SEEK_SET);

    // Read each data point
    for (int i = 0; i < num_data_points; i++) {
        for (int j = 0; j < 28*28; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, file);

            // Normalize to 0.0 - 1.0
            data[i][j] = pixel / 255.0;
        }
    }

    fclose(file);
    return 1;
}

// data array is assumed to be pre allocated matrix with num_data_points rows and 10 columns, 
// with all entries being zero. 
int dh_read_mnist_digits_labels(const char *filename, int num_data_points, float **data) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        return 0; // File not found
    }

    // Skip the header
    fseek(file, 8, SEEK_SET);

    // Read each label
    for (int i = 0; i < num_data_points; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);

        // One-hot encode label
        data[i][label] = 1.0;
    }

    fclose(file);
    return 1;
}