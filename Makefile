.PHONY: all clean benchmarks directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared --generate-code arch=compute_30,code=[compute_30,sm_30] \
	--generate-code arch=compute_35,code=[compute_35,sm_35] \
	--generate-code arch=compute_50,code=[compute_50,sm_50] \
	--generate-code arch=compute_53,code=[compute_53,sm_53]

all: directories benchmarks bin/runner

benchmarks: bin/mandelbrot.so bin/timer_spin.so bin/multikernel.so \
	bin/cpu_inorder_walk.so bin/cpu_random_walk.so bin/inorder_walk.so \
	bin/random_walk.so

directories:
	mkdir -p bin/
	mkdir -p obj/

bin/mandelbrot.so: src/mandelbrot.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/mandelbrot.so src/mandelbrot.cu

bin/timer_spin.so: src/timer_spin.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/timer_spin.so src/timer_spin.cu

bin/multikernel.so: src/multikernel.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/multikernel.so src/multikernel.cu

bin/inorder_walk.so: src/inorder_walk.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/inorder_walk.so src/inorder_walk.cu

bin/random_walk.so: src/random_walk.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/random_walk.so src/random_walk.cu

bin/cpu_inorder_walk.so: src/cpu_inorder_walk.c src/library_interface.h
	gcc $(CFLAGS) -shared -o bin/cpu_inorder_walk.so src/cpu_inorder_walk.c

bin/cpu_random_walk.so: src/cpu_random_walk.c src/library_interface.h
	gcc $(CFLAGS) -shared -o bin/cpu_random_walk.so src/cpu_random_walk.c

obj/cjson.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cjson.o src/third_party/cJSON.c

obj/parse_config.o: src/parse_config.c src/parse_config.h \
		src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/runner.o: src/runner.c src/third_party/cJSON.h src/library_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

obj/gpu_utilities.o: src/gpu_utilities.cu src/gpu_utilities.h \
		src/library_interface.h
	nvcc -c $(NVCCFLAGS) -o obj/gpu_utilities.o src/gpu_utilities.cu

bin/runner: obj/runner.o obj/cjson.o obj/parse_config.o obj/gpu_utilities.o
	nvcc $(NVCCFLAGS) -o bin/runner obj/runner.o obj/parse_config.o \
		obj/cjson.o obj/gpu_utilities.o -lpthread -ldl -lm

clean:
	rm -f bin/*
	rm -f obj/*
