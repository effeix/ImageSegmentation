CFLAGS = --std=c++11 -O0 -g

all: edge_filter imseg_s imseg_p

imseg_s: edge_filter ../main_seq.cpp ../imagem.cpp
	nvcc -o $@ ../main_seq.cpp ../imagem.cpp $(CFLAGS)

imseg_p: edge_filter ../main_par.cu ../imagem.cpp
	nvcc -o $@ ../main_par.cu ../imagem.cpp $(CFLAGS) -lnvgraph

edge_filter: ../edge.cu ../imagem.cpp
	nvcc -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f edge_filter imseg_p imseg_s
