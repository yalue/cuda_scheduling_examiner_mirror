.PHONY: all clean benchmarks directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

NVCCFLAGS := -arch=sm_50 -g --ptxas-options=-v --cudart shared \
	--compiler-options="$(CFLAGS)"

all: directories benchmarks bin/runner

benchmarks: bin/mandelbrot.so bin/timer_spin.so

directories:
	mkdir -p bin/
	mkdir -p obj/

obj/mandelbrot.o: src/mandelbrot.cu src/library_interface.h
	nvcc -c $(NVCCFLAGS) -o obj/mandelbrot.o src/mandelbrot.cu

bin/mandelbrot.so: obj/mandelbrot.o
	g++ -shared -o bin/mandelbrot.so obj/mandelbrot.o

obj/timer_spin.o: src/timer_spin.cu src/library_interface.h
	nvcc -c $(NVCCFLAGS) -o obj/timer_spin.o src/timer_spin.cu

bin/timer_spin.so: obj/timer_spin.o
	g++ -shared -o bin/timer_spin.so obj/timer_spin.o

obj/cjson.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cjson.o src/third_party/cJSON.c

obj/parse_config.o: src/parse_config.c src/parse_config.h \
		src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/runner.o: src/runner.c src/third_party/cJSON.h src/library_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

bin/runner: obj/runner.o obj/cjson.o obj/parse_config.o
	gcc $(CFLAGS) -o bin/runner obj/runner.o obj/cjson.o obj/parse_config.o \
		-lpthread -ldl

clean:
	rm -f bin/*
	rm -f obj/*
