CC := gcc-14
CCFLAGS := -O3 -L/opt/homebrew/opt/libomp/lib -framework Accelerate
# CCFLAGS := -g -L/opt/homebrew/opt/libomp/lib -framework Accelerate
SOURCE_FILES := src/phi_layers.c src/phi_utils.c src/ops_cpu.c
INCLUDE_FLAGS := -Iinclude -I/opt/homebrew/opt/libomp/include


all: test 

lib: $(SOURCE_FILES)
	mkdir -p build
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) -c $(SOURCE_FILES) 
	mv *.o build
	cd build && ar rcs libphi.a *.o

test: $(SOURCE_FILES) src/test.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/test.c -o bin/test

generate: $(SOURCE_FILES) src/generate.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/generate.c -o bin/gen

test_correctness: test
	rm -rf test_data
	mkdir test_data
	./bin/test && pytest

small_test: $(SOURCE_FILES) src/small_test.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/small_test.c -o bin/small_test

bench: bench_ops
	./bin/bench_ops

bench_ops: $(SOURCE_FILES) src/bench_ops.c
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/bench_ops.c -o bin/bench_ops

bin:
	mkdir bin

clean:
	rm -rf bin

