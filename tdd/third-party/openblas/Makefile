
all:
	tar zxf OpenBLAS-0.3.27.tar.gz 
	make -C OpenBLAS-0.3.27 FC=gfortran -j

install:
	make -C OpenBLAS-0.3.27 install PREFIX=../local/

clean:
	-rm -rf ./local/ ./OpenBLAS-0.3.27/
