all: matrixMulNaive

matrixMulMine: matrixMulMine.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

matrixMulMine.o: matrixMulMine.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

matrixMulNaive: matrixMulNaive.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

matrixMulNaive.o: matrixMulNaive.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

clean:
	rm -rf *.o
	rm matrixMulMine
	rm matrixMulNaive
