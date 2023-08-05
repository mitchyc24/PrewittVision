CC=gcc
NVCC=nvcc
CFLAGS=-I.
DEPS = kernels.h
OBJ = main.o grayscale.o prewitt.o
LIBS=-llodepng

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(CFLAGS)

PrewittVision: $(OBJ)
	$(NVCC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o PrewittVision
