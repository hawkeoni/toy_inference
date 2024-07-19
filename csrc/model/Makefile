SOURCE_FILES := src/phi_layers.c src/utils.c
INCLUDE_FLAGS := -Iinclude
test: $(SOURCE_FILES) src/test.c bin
	gcc $(INCLUDE_FLAGS) -g $(SOURCE_FILES) src/test.c -o bin/test

bin:
	mkdir bin

clean:
	rm -rf bin

