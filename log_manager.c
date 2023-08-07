#include <stdio.h>
#include <stdlib.h>

#define VERSION 0.1

// Structure to hold log data
typedef struct {

    float kernel_time_grayscale;
    float kernel_time_prewitt;

} LogData;

// Function to write log data
void write_log(LogData *data) {
    FILE *file = fopen("timing_log.txt", "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening log file.\n");
        return;
    }

    // Write the data to the file
    fprintf(file, "Version: %.1f\n", VERSION);
    fprintf(file, "Grayscale Kernel Time (ms): %f\n", data->kernel_time_grayscale);
    fprintf(file, "Prewitt Kernel Time (ms): %f\n", data->kernel_time_prewitt);
    fprintf(file, "----------------------------\n");

    fclose(file);
}
