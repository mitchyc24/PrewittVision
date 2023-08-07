#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

// Structure to hold log data
typedef struct {
    float kernel_time_grayscale;
    float kernel_time_prewitt;
} LogData;

// Function to write log data
void write_log(LogData *data);

#endif // LOG_MANAGER_H
