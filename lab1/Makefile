all: 1dStencil_1 1dStencil_2

1dStencil_1: 1dStencil_1.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

# It MUST be named .cu or nvcc compiles as regular C !!! (no __global__)
1dStencil_1.o: 1dStencil_1.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

1dStencil_2: 1dStencil_2.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

# It MUST be named .cu or nvcc compiles as regular C !!! (no __global__)
1dStencil_2.o: 1dStencil_2.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

clean:
	rm -rf *.o
	rm 1dStencil_1
	rm 1dStencil_2
