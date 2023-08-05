CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-I.
CUDAFLAGS=-I. -L/usr/local/cuda/lib64 -lcudart -ccbin $(CXX) 
DEPS = kernels.h lodepng.h
OBJ = main.o grayscale.o prewitt.o lodepng.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

PrewittVision: $(OBJ)
	$(NVCC) -o $@ $^ $(CFLAGS) $(CUDAFLAGS)

.PHONY: clean

clean:
	rm -f *.o PrewittVision
