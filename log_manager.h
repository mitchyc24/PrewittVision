#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#pragma once
#include <stdio.h>

typedef struct 
{
    FILE* file;
} LogFile;


typedef struct LogData {

    char img_name[256];
    float kernel_time_grayscale;
    float kernel_time_prewitt;
    struct LogData* next; // Pointer to the next log data

} LogData;

// Additional function to write all log data at once
void write_log(LogFile logFile, LogData* head);

LogFile open_log_file();
void free_log(LogData* head);


#endif // LOG_MANAGER_H
