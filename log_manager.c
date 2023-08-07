#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "log_manager.h"

#define VERSION 0.1


LogFile open_log_file() {
    LogFile logFile;
    logFile.file = fopen("timing_log.txt", "a");
    if (logFile.file == NULL) {
        fprintf(stderr, "Error opening log file.\n");
        logFile.file = NULL;
    }
    return logFile;
}

// Function to write all log data at once
void write_log(LogFile logFile, LogData* head) {
    if (logFile.file == NULL) {
        fprintf(stderr, "Log file is not open.\n");
        return;
    }

    fprintf(logFile.file, "Version: %.1f\n", VERSION);

    LogData* current = head;
    while (current != NULL) {

        fprintf(logFile.file, "Image: %s\n", current->img_name);
        fprintf(logFile.file, "\tGrayscale Kernel Time (ms): %f\n", current->kernel_time_grayscale);
        fprintf(logFile.file, "\tPrewitt Kernel Time (ms): %f\n", current->kernel_time_prewitt);
        
        current = current->next; // Move to the next log data
    }

    fprintf(logFile.file, "----------------------------\n");
}


void free_log(LogData* head){
    // Free the linked list
    LogData* current = head;
    while (current != NULL) {
        LogData* temp = current;
        current = current->next;
        free(temp);
    }
}