#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "log_manager.h"

#define VERSION 0.1


FILE* open_file(char* filename) {
    FILE* file;
    file = fopen(filename, "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        file = NULL;
    }
    return file;
}



// Function to write all log data at once
void write_log(FILE* file, LogData* head) {
    if (file == NULL) {
        fprintf(stderr, "Log file is not open.\n");
        return;
    }

    fprintf(file, "Version: %.1f\n", VERSION);

    LogData* current = head;
    while (current != NULL) {

        fprintf(file, "Image: %s\n", current->img_name);
        fprintf(file, "\tBlock Size: %d\n", current->block_size);
        fprintf(file, "\tGrayscale Kernel Time (ms): %f\n", current->kernel_time_grayscale);
        fprintf(file, "\tPrewitt Kernel Time (ms): %f\n", current->kernel_time_prewitt);
        
        current = current->next; // Move to the next log data
    }

    fprintf(file, "----------------------------\n");
}

void write_log_to_csv(FILE* file, LogData* head) {
    if (file == NULL) {
        fprintf(stderr, "Log file is not open.\n");
        return;
    }

    // Print the CSV header
    fprintf(file, "Image,Grayscale Kernel Time (ms),Prewitt Kernel Time (ms),Block Size\n");

    LogData* current = head;
    while (current != NULL) {
        fprintf(file, "%s,%.3f,%.3f,%d\n", current->img_name, current->kernel_time_grayscale, current->kernel_time_prewitt, current->block_size);
        current = current->next; // Move to the next log data
    }
}


void free_log_data(LogData* head){
    // Free the linked list
    LogData* current = head;
    while (current != NULL) {
        LogData* temp = current;
        current = current->next;
        free(temp);
    }
}