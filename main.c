#include "kernels.h"
#include "log_manager.h"
#include "lodepng.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>

int main(int argc, char* argv[]) {
    DIR *dir;
    struct dirent *entry;
    char path[] = "imgs"; // Path to the directory containing images
    char filenames[512][512]; // Array to store filenames
    int num_images = 0;

    dir = opendir(path);
    if (dir == NULL) {
        fprintf(stderr, "Could not open directory %s\n", path);
        return 1;
    }

    // Collect the filenames from path directory
    while ((entry = readdir(dir)) != NULL) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/%s", path, entry->d_name);
        struct stat st;
        if (stat(filename, &st) == 0 && S_ISREG(st.st_mode)) {
            strcpy(filenames[num_images], filename);
            num_images++;
        }
    }
    closedir(dir);

    // Open log files to write
    FILE* logFile = open_file("timing_log.txt"); // Open txt log file
    FILE* csvFile = open_file("timing_log.csv"); // OPEN csv log file

    // Store input/output data in linked list
    LogData* head = NULL;
    LogData* tail = NULL;

    // Loop through each image within the path directory
    // #pragma omp parallel for // uncomment this and comment out line 100-120 to run images in parallel
    for (int i = 0; i < num_images; i++) {
        //int thread_id = omp_get_thread_num();
        //printf("Processing image %d by thread %d\n", i, thread_id);

        char *filename = filenames[i];

        unsigned char* host_input_image;
        unsigned int width, height;
        
        // Decode the PNG into an array of char
        if (lodepng_decode32_file(&host_input_image, &width, &height, filename)) {
            fprintf(stderr, "Error reading image file %s\n", filename);
            free(host_input_image);
            continue; // Skip to the next file
        }

        // Create and allocate memory for new images
        unsigned char* host_output_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
        unsigned char* host_grayscale_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));

        LogData* log_data = (LogData*)malloc(sizeof(LogData));
        strncpy(log_data->img_name, filename, sizeof(log_data->img_name) - 1);
        log_data->img_name[sizeof(log_data->img_name) - 1] = '\0'; // Null-terminate the string


        int block_sizes[6] = { 4, 8, 16, 32, 64, 128 };
        int num_of_threads[4] = { 1, 2, 6, 16 };
        LogData* current = NULL;

        // Loop through each block size for CUDA processing
        for (int i = 0; i < 6; i++) {
            int b_size = block_sizes[i];
            LogData* log_data = (LogData*)malloc(sizeof(LogData));
            strcpy(log_data->type, "CUDA");
            strncpy(log_data->img_name, filename, sizeof(log_data->img_name) - 1);
            log_data->img_name[sizeof(log_data->img_name) - 1] = '\0';
            log_data->block_size = b_size;
            log_data->width = width;
            log_data->height = height;
            log_data->kernel_time_grayscale = convertToGrayscaleCuda(host_input_image, host_grayscale_image, width, height, b_size);
            log_data->kernel_time_prewitt = applyPrewittCuda(host_grayscale_image, host_output_image, width, height, b_size);
            log_data->next = NULL;

            if (head == NULL) {
                head = log_data;
                tail = log_data;
            } else {
                tail->next = log_data;
                tail = log_data;
            }
        }

        // Loop through each number of threads by CPU/OpenMP
        for (int i = 0; i < 4; i++) {
            int threads = num_of_threads[i];
            LogData* log_data = (LogData*)malloc(sizeof(LogData));
            strcpy(log_data->type, "OpenMP");
            strncpy(log_data->img_name, filename, sizeof(log_data->img_name) - 1);
            log_data->img_name[sizeof(log_data->img_name) - 1] = '\0';
            log_data->block_size = threads;
            log_data->width = width;
            log_data->height = height;
            log_data->kernel_time_grayscale = convertToGrayscaleThreading(host_input_image, host_grayscale_image, width, height, threads);
            log_data->kernel_time_prewitt = applyPrewittThreading(host_grayscale_image, host_output_image, width, height, threads);
            log_data->next = NULL;

            if (head == NULL) {
                head = log_data;
                tail = log_data;
            } else {
                tail->next = log_data;
                tail = log_data;
            }
        }

        // Save converted images (greyscale and edge detected with Prewitt operator)
        char base_name_copy[256];
        const char* base_name = extract_base_name(filename, base_name_copy, sizeof(base_name_copy));

        encode_and_save("grayscale", base_name, host_grayscale_image, width, height);
        encode_and_save("prewitt", base_name, host_output_image, width, height);

        free(host_input_image);
        free(host_output_image);
        free(host_grayscale_image);
    }

    write_log(logFile, head); 
    fclose(logFile); // Close the log file

    write_log_to_csv(csvFile, head);
    fclose(csvFile);

    free_log_data(head);


    return 0;
}
