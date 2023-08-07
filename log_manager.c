#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define VERSION 0.1

// Structure to hold the file
typedef struct 
{
    FILE* file;
} LogFile;

// Structure to hold log data
typedef struct {
    float kernel_time_grayscale;
    float kernel_time_prewitt;
} LogData;

LogFile open_log_file() {
    LogFile logFile;
    logFile.file = fopen("timing_log.txt", "a");
    if (logFile.file == NULL) {
        fprintf(stderr, "Error opening log file.\n");
        logFile.file = NULL;
    }
    return logFile;
}

// Function to write log data
void write_log(LogFile logFile, LogData *data) {
    if (logFile.file == NULL) {
        fprintf(stderr, "Log file is not open.\n");
        return;
    }
    
    #pragma omp critical
    {
        fprintf(logFile.file, "Version: %.1f\n", VERSION);
        fprintf(logFile.file, "Grayscale Kernel Time (ms): %f\n", data->kernel_time_grayscale);
        fprintf(logFile.file, "Prewitt Kernel Time (ms): %f\n", data->kernel_time_prewitt);
        fprintf(logFile.file, "----------------------------\n");

    }
}

