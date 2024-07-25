CC := gcc-14
CCFLAGS := -fopenmp -O2
SOURCE_FILES := src/phi_layers.c src/phi_utils.c src/ops_cpu.c
INCLUDE_FLAGS := -Iinclude


all: test

test: $(SOURCE_FILES) src/test.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/test.c -o bin/test

small_test: $(SOURCE_FILES) src/small_test.c bin
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/small_test.c -o bin/small_test

bench_ops: $(SOURCE_FILES) src/bench_ops.c
	$(CC) $(CCFLAGS) $(INCLUDE_FLAGS) $(SOURCE_FILES) src/bench_ops.c -o bin/bench_ops

bin:
	mkdir bin

clean:
	rm -rf bin

