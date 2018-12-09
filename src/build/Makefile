CFLAGS = --std=c++11 -O0 -g

all: edge_filter imseg_s imseg_p

imseg_s: ../main_seq.cpp ../imagem.cpp
	nvcc -o $@ $^ $(CFLAGS)

imseg_p: ../main_par.cu ../imagem.cpp
	nvcc -o $@ $^ $(CFLAGS) -lnvgraph

edge_filter: ../edge.cu ../imagem.cpp
	nvcc -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f edge_filter imseg_p imseg_s
