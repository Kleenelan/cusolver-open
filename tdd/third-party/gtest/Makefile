
GTEST := googletest-release-1.10.0

all:
	tar zxf $(GTEST).tar.gz
	export CXXFLAGS="-Wno-error=maybe-uninitialized"
	cmake -S $(GTEST)/ -B ./build/
	make -j -C build/

.PHONY: clean
clean:
	-rm -rf $(GTEST) ./build/



