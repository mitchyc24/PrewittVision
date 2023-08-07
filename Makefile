CC=gcc
NVCC=nvcc

CFLAGS=-I.
NVFLAGS=-I. -L/usr/local/cuda/lib64 -lcudart -ccbin g++

DEPS = kernels.h lodepng.h utils.h
OBJ = main.o grayscale.o prewitt.o lodepng.o utils.o timing.o log_manager.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVFLAGS)

PrewittVision: $(OBJ)
	$(NVCC) -o $@ $^ $(NVFLAGS)

.PHONY: clean

clean:
	rm -f $(OBJ) PrewittVision
