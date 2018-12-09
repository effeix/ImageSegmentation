# ImageSegmentation
Software para segmentação de imagens com paralelização em GPU

## Objetivo
Neste projeto o objetivo é realizar a separação de um objeto seleciona como plano de frente do resto da imagem (plano de fundo). Esta técnica é conhecida como segmentação de imagens. O usuário deve escolher alguns pontos da imagem dentro do objeto que deseja selecionar e alguns pontos dentro dos objetos que deseja ignorar. A imagem final será uma máscara em preto e branco representando esta separação que, ao ser multiplicada com a imagem original, gera a nova imagem segmentada. A parte final (multiplicação) não está implementada no momento. Existem duas versões do programa: uma sequencial e uma paralela, sendo que a segunda utiliza estratégias de paralelização em GPU com CUDA (Nvidia).

## Explicação
Primeiramente, devemos interpretar a imagem a ser segmentada como um grafo (conjunto de nós que possuem conexões entre eles). Neste grafo cada pixel da imagem é um nó e possue quatro nós vizinhos: os pixels de cima, de baixo da esquerda e da direita na imagem original. Ligando estes nós existem arestas cujo peso é a diferença de cor entre os dois nós ligados por ela. Por exemplo: se temos um nó `i` de cor 255 (branco) ligado à um nó `j` de cor 127 (cinza), a aresta que liga estes dois nós tem peso 255 - 127 = 128. Depois de calculados todos os pesos das arestas, deseja-se descobrir, para cada pixel, se este está mais perto de uma semente de frente ou uma de fundo. Para isto, deve-se achar o menor caminho (com menor soma de pesos) entre um pixel `i` e as sementes de fundo e o menor caminho entre o mesmo pixel `i` e as sementes de frente. Dentre estes menores caminhos, o menor deles mostra a qual plano o pixel pertence. O algoritmo utilizado para encontrar o menor caminho entre dois pixels foi o SSSP (Single Source Shortest Path). 

## Dependências
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Thrust](http://thrust.github.io/)
- [Python 3.6.5+](https://www.python.org/)
- Make

## Utilização
Primeiramente, deve-se compilar os três executáveis utilizados pelo programa. Vá para a pasta `ImageSegmentation/src/build/` e execute o comando `make`.

Com os executáveis criados já é possível utilizar o software da seguinte maneira:
```sh
$ ./imseg_p <imagem_entrada> <imagem_saida> < sementes.txt
```
para a versão paralela, ou
```sh
$ ./imseg_s <imagem_entrada> <imagem_saida> < sementes.txt
```
para a versão sequencial. Caso não possua uma placa de vídeo (GPU) da marca Nvidia, a versão paralela não funcionará.

O arquivo `sementes.txt` (pode ter qualquer nome) contém as sementes de frente e fundo no seguinte formato:

```
N_FG N_BG
X1 Y1
X2 Y2
X3 Y3
...
```
em que N_FG é a quantidade de sementes de frente, N_BG é a quantidade de sementes de fundo e logo abaixo coordenadas dos pixel de frente e fundo (N_FG coordenadas e em seguida N_BG coordenadas).

Foram observados resultados mais satisfatórios quando as imagens de entrada eram um filtro de bordas da imagem original, ou seja, uma nova imagem destacando somente os contornos. Portanto é recomendado que a imagem de entrada dos executáveis `imseg_p` e `imseg_s` seja uma imagem filtrada.

**Filtro de bordas na imagem**
Dentro da pasta `build/` execute:
```sh
$ ./edge_filter <imagem_entrada> <imagem_saida>
```
Isto gerará uma imagem filtrada.

**Uma limitação do projeto é o suporte apenas à imagens do tipo PGM no formato ASCII. Este tipo pode ser obtido convertendo uma imagem qualquer (PNG, JPG, etc) em PGM utilizando o software GIMP.**

## Testes
Para verificar a performance do programa, algumas métricas de tempo foram utilizadas. Foram medidos os tempos de criação do grafo (GRAPH), execução do algoritmo SSSP (SSSP) e criação da máscara final (IMAGE). O programa imprime, no final da execução, os tempos medidos e o tempo total no seguinte formato:

```
TIMING
GRAPH: V ms
SSSP: X ms
IMAGE: Y ms
TOTAL: Z ms
```
Na versão sequencial, a criação do grafo e execução do algoritmo são realizadas ao mesmo tempo, portanto o tempo GRAPH não aparece. Os testes foram realizados com versões filtradas das imagens. As imagens de teste se encontram em `ImageSegmentation/test/`, sendo que cada imagem possue a versão original (.pgm), a versão filtrada (com sufixo \_edge) e versão final (com sufixo \_out) dentro de suas respectivas pastas, bem como o arquivo sementes.txt utilizado.

#### Imagem 1: test/balls/balls.pgm (500x500 - 972Kb)

| Tempos (ms) | Sequencial | Paralela |
|:-----------:|------------|----------|
| GRAPH       | -          | 114.017  |
| SSSP        | 569.091    | 3389.42  |
| IMAGE       | 15.8321    | 17.0986  |
| TOTAL       | 587.2      | 3522.75  |

#### Imagem 2: test/stickers/stickers.pgm (1000x1000 - 3.7Mb)

| Tempos (ms) | Sequencial | Paralela |
|:-----------:|------------|----------|
| GRAPH       | -          | 493.55   |
| SSSP        | 2847.61    | 14258  	|
| IMAGE       | 6808.94    | 70.3044	|
| TOTAL       | 2919.03    | 14828.1  |

#### Imagem 3: test/logo/logo.pgm (2000x2000 - 14Mb)

| Tempos (ms) | Sequencial | Paralela |
|:-----------:|------------|----------|
| GRAPH       | -          | 0.001824 |
| SSSP        | 11203.9    | 189854  	|
| IMAGE       | 260.536    | 261.807 	|
| TOTAL       | 11464.5    | 195576   |
