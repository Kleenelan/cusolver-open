


EXE := 	hello_gtest \
	hello_cuda \
	gtest_cusolverDnSgeqrf \
	gtest_cusolverDnXgeqrf
	

all: $(EXE)

INC := -I./third-party/gtest/googletest-release-1.10.0/googletest/include -I/usr/local/cuda/include -I./include
LD_FLAGS :=  -L third-party/gtest/build/lib/ -lgtest -lgtest_main -L /usr/local/cuda/lib64 -lcudart -lcublas -lcusolver

%: %.cpp
	g++ -g $< -o $@  $(INC) $(LD_FLAGS)

.PHONY: clean
clean:
	-rm -rf $(EXE)
