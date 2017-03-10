.PHONY: all clean benchmarks directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -arch=sm_50 -g --ptxas-options=-v --compiler-options="$(CFLAGS)" \
	--cudart=shared

all: directories benchmarks bin/runner

benchmarks: bin/mandelbrot.so bin/timer_spin.so

directories:
	mkdir -p bin/
	mkdir -p obj/

bin/mandelbrot.so: src/mandelbrot.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/mandelbrot.so src/mandelbrot.cu

bin/timer_spin.so: src/timer_spin.cu src/library_interface.h
	nvcc --shared $(NVCCFLAGS) -o bin/timer_spin.so src/timer_spin.cu

obj/cjson.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cjson.o src/third_party/cJSON.c

obj/parse_config.o: src/parse_config.c src/parse_config.h \
		src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/runner.o: src/runner.c src/third_party/cJSON.h src/library_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

bin/runner: obj/runner.o obj/cjson.o obj/parse_config.o
	nvcc $(NVCCFLAGS) -o bin/runner obj/runner.o obj/cjson.o obj/parse_config.o \
		-lpthread -ldl -lm

clean:
	rm -f bin/*
	rm -f obj/*
