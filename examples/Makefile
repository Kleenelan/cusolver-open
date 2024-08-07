
EXE := ex_cusolverDnsgetrf_testing

UTILS_SRC := ../utils/cusolver_env.cpp \
			 ../utils/multi_thread_mt19937_64_rand.cpp


all: $(EXE)

#%: %.cpp $(UTILS_SRC) ../tdd/third-party/openblas/local/lib/libopenblas.a
%: %.cpp $(UTILS_SRC)
	g++ -fopenmp $^  -o $@ $(INC) $(LD_FLAGS)

INC := -I/usr/local/cuda/include -I ../utils/inc/
INC += -I../tdd/third-party/openblas/local/include/
LD_FLAGS := -L/usr/local/cuda/lib64 -lcudart -lcusolver
#LD_FLAGS += -L../tdd/third-party/lapack/local/lib -llapack  -lrefblas  -ltmglib -lgfortran
LD_FLAGS += -L../tdd/third-party/openblas/local/lib -lopenblas


.PHONY: clean
clean:
	-rm -rf $(EXE)
