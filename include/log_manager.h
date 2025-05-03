#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#pragma once
#include <stdio.h>


typedef struct LogData {
    char img_name[256];
    int block_size; 
    float kernel_time_grayscale;
    float kernel_time_prewitt;
    struct LogData* next; // Pointer to the next log data
} LogData;

FILE* open_file(char* filename);

// Additional function to write all log data at once
void write_log(FILE* logFile, LogData* head);
void write_log_to_csv(FILE* csvFile, LogData* head);

void free_log_data(LogData* head);


#endif // LOG_MANAGER_H
