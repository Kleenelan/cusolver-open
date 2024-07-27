
GTEST := googletest-release-1.10.0

export CXXFLAGS="-Wno-error=maybe-uninitialized"

all:
	cmake -S $(GTEST)/ -B ./build/
	make -j -C build/

.PHONY: clean
clean:
	-rm -rf ./build/



