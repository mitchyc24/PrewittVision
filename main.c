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

    // Collect the filenames
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

    LogFile logFile = open_log_file(); // Open the log file

    #pragma omp parallel for
    for (int i = 0; i < num_images; i++) {
        int thread_id = omp_get_thread_num();
        printf("Processing image %d by thread %d\n", i, thread_id);

        char *filename = filenames[i];

        unsigned char* host_input_image;
        unsigned int width, height;
        LogData log_data;

        if (lodepng_decode32_file(&host_input_image, &width, &height, filename)) {
            fprintf(stderr, "Error reading image file %s\n", filename);
            free(host_input_image);
            continue; // Skip to the next file
        }

        unsigned char* host_output_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
        unsigned char* host_grayscale_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));

        log_data.kernel_time_grayscale = convertToGrayscale(host_input_image, host_grayscale_image, width, height);

        char base_name_copy[256];
        const char* base_name = extract_base_name(filename, base_name_copy, sizeof(base_name_copy));

        encode_and_save("grayscale", base_name, host_grayscale_image, width, height);

        log_data.kernel_time_prewitt = applyPrewitt(host_grayscale_image, host_output_image, width, height);

        encode_and_save("prewitt", base_name, host_output_image, width, height);

        write_log(logFile, &log_data); // Write log data using the LogFile object

        free(host_input_image);
        free(host_output_image);
        free(host_grayscale_image);
    }

    fclose(logFile.file); // Close the log file

    return 0;
}
