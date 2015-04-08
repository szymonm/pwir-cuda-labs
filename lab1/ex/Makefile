all: 2dStencil

2dStencil: 2dStencil.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

# It MUST be named .cu or nvcc compiles as regular C !!! (no __global__)
2dStencil.o: 2dStencil.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<

clean:
	rm -rf *.o
	rm 2dStencil
