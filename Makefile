.PHONY: all clean benchmarks directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared --generate-code arch=compute_30,code=[compute_30,sm_30] \
	--generate-code arch=compute_35,code=[compute_35,sm_35] \
	--generate-code arch=compute_50,code=[compute_50,sm_50] \
	--generate-code arch=compute_53,code=[compute_53,sm_53] \
	--generate-code arch=compute_60,code=[compute_60,sm_60] \
	--generate-code arch=compute_62,code=[compute_62,sm_62]

BENCHMARK_DEPENDENCIES := src/library_interface.h \
	src/benchmark_gpu_utilities.h obj/benchmark_gpu_utilities.o

all: directories benchmarks bin/runner

benchmarks: bin/mandelbrot.so bin/timer_spin.so bin/multikernel.so \
	bin/cpu_inorder_walk.so bin/cpu_random_walk.so bin/inorder_walk.so \
	bin/random_walk.so bin/sharedmem_timer_spin.so bin/counter_spin.so \
	bin/timer_spin_default_stream.so bin/stream_action.so

visionworks: obj/cjson.o bin/runner
	$(MAKE) -C ./src/third_party/VisionWorks-1.6-Demos

directories:
	mkdir -p bin/
	mkdir -p obj/

bin/mandelbrot.so: src/mandelbrot.cu $(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/mandelbrot.so src/mandelbrot.cu \
		obj/benchmark_gpu_utilities.o

bin/timer_spin.so: src/timer_spin.cu $(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/timer_spin.so src/timer_spin.cu \
		obj/benchmark_gpu_utilities.o

bin/counter_spin.so: src/counter_spin.cu $(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/counter_spin.so src/counter_spin.cu \
		obj/benchmark_gpu_utilities.o

bin/timer_spin_default_stream.so: src/timer_spin_default_stream.cu \
		$(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/timer_spin_default_stream.so \
		src/timer_spin_default_stream.cu obj/benchmark_gpu_utilities.o

bin/multikernel.so: src/multikernel.cu $(BENCHMARK_DEPENDENCIES) obj/cjson.o
	nvcc --shared $(NVCCFLAGS) -o bin/multikernel.so src/multikernel.cu \
		obj/benchmark_gpu_utilities.o obj/cjson.o

bin/inorder_walk.so: src/inorder_walk.cu $(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/inorder_walk.so src/inorder_walk.cu \
		obj/benchmark_gpu_utilities.o

bin/random_walk.so: src/random_walk.cu $(BENCHMARK_DEPENDENCIES)
	nvcc --shared $(NVCCFLAGS) -o bin/random_walk.so src/random_walk.cu \
		obj/benchmark_gpu_utilities.o

bin/sharedmem_timer_spin.so: src/sharedmem_timer_spin.cu \
		$(BENCHMARK_DEPENDENCIES) obj/cjson.o
	nvcc --shared $(NVCCFLAGS) -o bin/sharedmem_timer_spin.so \
		src/sharedmem_timer_spin.cu obj/benchmark_gpu_utilities.o obj/cjson.o

bin/stream_action.so: src/stream_action.cu $(BENCHMARK_DEPENDENCIES) \
		obj/cjson.o
	nvcc --shared $(NVCCFLAGS) -o bin/stream_action.so src/stream_action.cu \
		obj/benchmark_gpu_utilities.o obj/cjson.o

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

obj/benchmark_gpu_utilities.o: src/benchmark_gpu_utilities.cu \
		src/benchmark_gpu_utilities.h
	nvcc -c $(NVCCFLAGS) -o obj/benchmark_gpu_utilities.o \
		src/benchmark_gpu_utilities.cu

obj/barrier_wait.o: src/barrier_wait.c src/barrier_wait.h
	gcc -c $(CFLAGS) -o obj/barrier_wait.o src/barrier_wait.c

bin/runner: obj/runner.o obj/cjson.o obj/parse_config.o obj/gpu_utilities.o \
		obj/barrier_wait.o
	nvcc $(NVCCFLAGS) -o bin/runner obj/runner.o obj/parse_config.o \
		obj/cjson.o obj/gpu_utilities.o obj/barrier_wait.o -lpthread -ldl -lm

clean:
	rm -f bin/*
	rm -f obj/*
