.PHONY: all clean benchmarks directories

CFLAGS := -Wall -Werror -O3 -g -fPIC
NVCC ?= /usr/local/cuda/bin/nvcc
NVCCFLAGS := -g --ptxas-options=-v --compiler-options="$(CFLAGS)"

# There was no easy way to autodetect what targets NVCC supported prior to CUDA
# 11.5, so a lookup table is hardcoded below.
# ***If a target is listed below, that does not mean we support it.***
CUDA_VER=$(shell $(NVCC) --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$$/\1/p' | tr -d '.')

# Tesla supported up through CUDA 6.5
ifeq ($(shell expr \( $(CUDA_VER) \>= 55 \) \& \( $(CUDA_VER) \<= 65 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_10,code=[compute_10,sm_10]
	NVCCFLAGS += --generate-code arch=compute_11,code=[compute_11,sm_11]
	NVCCFLAGS += --generate-code arch=compute_12,code=[compute_12,sm_12]
	NVCCFLAGS += --generate-code arch=compute_13,code=[compute_13,sm_13]
endif

# Fermi supported up through CUDA 8
ifeq ($(shell expr \( $(CUDA_VER) \>= 55 \) \& \( $(CUDA_VER) \<= 80 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_20,code=[compute_20,sm_20]
endif

# Kepler (first gen) supported up through CUDA 10
ifeq ($(shell expr \( $(CUDA_VER) \>= 55 \) \& \( $(CUDA_VER) \<= 100 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_30,code=[compute_30,sm_30]
	NVCCFLAGS += --generate-code arch=compute_32,code=[compute_32,sm_32]
endif

ifeq ($(shell expr \( $(CUDA_VER) \>= 55 \) \& \( $(CUDA_VER) \<= 115 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_35,code=[compute_35,sm_35]
endif

ifeq ($(shell expr \( $(CUDA_VER) \>= 70 \) \& \( $(CUDA_VER) \< 115 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_50,code=[compute_50,sm_50]
	NVCCFLAGS += --generate-code arch=compute_52,code=[compute_52,sm_52]
	NVCCFLAGS += --generate-code arch=compute_53,code=[compute_53,sm_53]
endif

ifeq ($(shell expr \( $(CUDA_VER) \>= 90 \) \& \( $(CUDA_VER) \< 115 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_60,code=[compute_60,sm_60]
	NVCCFLAGS += --generate-code arch=compute_61,code=[compute_61,sm_61]
	NVCCFLAGS += --generate-code arch=compute_62,code=[compute_62,sm_62]
	NVCCFLAGS += --generate-code arch=compute_70,code=[compute_70,sm_70]
endif

ifeq ($(shell expr \( $(CUDA_VER) \>= 100 \) \& \( $(CUDA_VER) \< 115 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_72,code=[compute_72,sm_72]
	NVCCFLAGS += --generate-code arch=compute_75,code=[compute_75,sm_75]
endif

ifeq ($(shell expr \( $(CUDA_VER) \>= 112 \) \& \( $(CUDA_VER) \< 115 \)), 1)
	NVCCFLAGS += --generate-code arch=compute_80,code=[compute_80,sm_80]
	NVCCFLAGS += --generate-code arch=compute_86,code=[compute_86,sm_86]
endif

# Starting with CUDA 11.5, nvcc supports auto-detection with -arch=all
ifeq ($(shell expr $(CUDA_VER) \>= 115), 1)
	NVCCFLAGS += --gpu-architecture all
endif

BENCHMARK_DEPENDENCIES := src/library_interface.h \
	src/benchmark_gpu_utilities.h obj/benchmark_gpu_utilities.o

all: directories benchmarks bin/runner

benchmarks: bin/mandelbrot.so bin/timer_spin.so bin/multikernel.so \
	bin/cpu_inorder_walk.so bin/cpu_random_walk.so bin/inorder_walk.so \
	bin/random_walk.so bin/sharedmem_timer_spin.so bin/counter_spin.so \
	bin/timer_spin_default_stream.so bin/stream_action.so bin/task_host.so \
	bin/matrix_multiply.so

directories:
	mkdir -p bin/
	mkdir -p obj/

bin/mandelbrot.so: src/mandelbrot.cu $(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/mandelbrot.so \
	src/mandelbrot.cu obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/timer_spin.so: src/timer_spin.cu $(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/timer_spin.so \
		src/timer_spin.cu obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/counter_spin.so: src/counter_spin.cu $(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/counter_spin.so \
		src/counter_spin.cu obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/timer_spin_default_stream.so: src/timer_spin_default_stream.cu \
		$(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o \
		bin/timer_spin_default_stream.so src/timer_spin_default_stream.cu \
		obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/multikernel.so: src/multikernel.cu $(BENCHMARK_DEPENDENCIES) obj/cjson.o
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/multikernel.so \
		src/multikernel.cu obj/benchmark_gpu_utilities.o obj/cjson.o $(LDLIBS)

bin/inorder_walk.so: src/inorder_walk.cu $(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/inorder_walk.so \
		src/inorder_walk.cu obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/random_walk.so: src/random_walk.cu $(BENCHMARK_DEPENDENCIES)
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/random_walk.so \
		src/random_walk.cu obj/benchmark_gpu_utilities.o $(LDLIBS)

bin/sharedmem_timer_spin.so: src/sharedmem_timer_spin.cu \
		$(BENCHMARK_DEPENDENCIES) obj/cjson.o
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/sharedmem_timer_spin.so \
		src/sharedmem_timer_spin.cu obj/benchmark_gpu_utilities.o obj/cjson.o \
		$(LDLIBS)

bin/stream_action.so: src/stream_action.cu $(BENCHMARK_DEPENDENCIES) \
		obj/cjson.o
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/stream_action.so \
		src/stream_action.cu obj/benchmark_gpu_utilities.o obj/cjson.o $(LDLIBS)

bin/matrix_multiply.so: src/matrix_multiply.cu $(BENCHMARK_DEPENDENCIES) \
		obj/cjson.o
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/matrix_multiply.so \
		src/matrix_multiply.cu obj/benchmark_gpu_utilities.o obj/cjson.o \
		$(LDLIBS)

bin/cpu_inorder_walk.so: src/cpu_inorder_walk.c src/library_interface.h
	gcc $(CFLAGS) $(LDFLAGS) -shared -o bin/cpu_inorder_walk.so \
		src/cpu_inorder_walk.c $(LDLIBS)

bin/cpu_random_walk.so: src/cpu_random_walk.c src/library_interface.h
	gcc $(CFLAGS) $(LDFLAGS) -shared -o bin/cpu_random_walk.so \
		src/cpu_random_walk.c $(LDLIBS)

obj/cjson.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cjson.o src/third_party/cJSON.c

obj/parse_config.o: src/parse_config.c src/parse_config.h \
		src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/runner.o: src/runner.c src/library_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

obj/task_host.o: src/task_host.c src/third_party/cJSON.h \
	src/library_interface.h
	gcc -c $(CFLAGS) -o obj/task_host.o src/task_host.c

obj/task_host_utilities.o: src/task_host_utilities.cu \
		src/task_host_utilities.h src/library_interface.h
	$(NVCC) -c $(NVCCFLAGS) -o obj/task_host_utilities.o \
		src/task_host_utilities.cu

obj/benchmark_gpu_utilities.o: src/benchmark_gpu_utilities.cu \
		src/benchmark_gpu_utilities.h
	$(NVCC) -c $(NVCCFLAGS) -o obj/benchmark_gpu_utilities.o \
		src/benchmark_gpu_utilities.cu

obj/barrier_wait.o: src/barrier_wait.c src/barrier_wait.h
	gcc -c $(CFLAGS) -o obj/barrier_wait.o src/barrier_wait.c

bin/runner: obj/runner.o bin/task_host.so
	gcc $(LDFLAGS) -o bin/runner obj/runner.o -ldl $(LDLIBS)

bin/task_host.so: obj/task_host.o obj/cjson.o obj/parse_config.o \
		obj/task_host_utilities.o obj/barrier_wait.o
	$(NVCC) $(LDFLAGS) --shared $(NVCCFLAGS) -o bin/task_host.so \
		obj/task_host.o obj/cjson.o obj/task_host_utilities.o \
		obj/barrier_wait.o obj/parse_config.o -lpthread -ldl -lm $(LDLIBS)

clean:
	rm -f bin/*
	rm -f obj/*
