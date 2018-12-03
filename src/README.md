Projeto 4: segmentação de imagens
======================


Este projeto contém uma implementação sequencial do algoritmo **Single Source Shortest Path** e sua utilização 
para a criação de uma segmentação de imagens em frente e fundo. Este programa não foi extensamente testado em 
imagens grandes ou complexas. 

**Limitações**:

1. o projeto só aceita uma semente de frente e uma semente fundo
1. o algoritmo **SSSP* é executado duas vezes (uma para frente e uma para fundo)
1. na função *SSSP* cada ponto é adicionado a fila de prioridade mais de uma vez, mas só é processado se houver chance de melhorar seu caminho. 
1. o leitor de PGM é extremamente limitado e não suporta comentários no PGM. O melhor é converter usando o programa `convert input.png -compress none -depth 8 out.pgm` 

**Testes iniciais**:

A imagem *teste.pgm* contém um exemplo simples para segmentação. Dois testes possíveis e interessantes são

1. colocar a semente dentro do círculo grande e a semente de fundo fora dos dois círculos
1. colocar a semente dentro do círculo grande e a semente de fundo dentro do pequeno. 

Antes de rodar, tente prever o comportamento da segmentação final. 

