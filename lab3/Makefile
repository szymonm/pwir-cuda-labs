all: independentSet

independentSet: independentSet.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

independentSet.o: independentSet.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

clean:
	rm -rf *.o
	rm independentSet
