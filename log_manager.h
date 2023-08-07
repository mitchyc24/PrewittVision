#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#pragma once
#include <stdio.h>

typedef struct 
{
    FILE* file;
} LogFile;


// Structure to hold log data
typedef struct {
    float kernel_time_grayscale;
    float kernel_time_prewitt;
} LogData;


LogFile open_log_file();
void write_log(LogFile logFile, LogData *data);

#endif // LOG_MANAGER_H
