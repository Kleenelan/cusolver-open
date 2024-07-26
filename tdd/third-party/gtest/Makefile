
GTEST := googletest-release-1.10.0

all:
	export CXXFLAGS="-Wno-error=maybe-uninitialized"
	cmake -S $(GTEST)/ -B ./build/
	make -j -C build/

.PHONY: clean
clean:
	-rm -rf ./build/



