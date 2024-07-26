CC := clang
CCFLAGS := -O3 -L/opt/homebrew/opt/libomp/lib -framework Accelerate
SOURCE_FILES := src/phi_layers.c src/phi_utils.c src/ops_cpu.c
INCLUDE_FLAGS := -Iinclude -I/opt/homebrew/opt/libomp/include


all: test 

test: $(SOURCE_FILES) src/test.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/test.c -o bin/test

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

