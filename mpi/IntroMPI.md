# Resumo do MPI e Exemplos Práticos

## Visão Geral

O MPI (Message Passing Interface) é uma especificação amplamente adotada para comunicação entre processos em ambientes paralelos e distribuídos. Sua principal vantagem é a capacidade de escalar aplicações científicas e de engenharia para milhares ou até milhões de núcleos de processamento. Com MPI, cada processo possui seu próprio espaço de memória e a comunicação é realizada via troca de mensagens explícitas, o que o diferencia de abordagens baseadas em memória compartilhada, como OpenMP.

MPI é uma biblioteca portátil e de alto desempenho, com implementações amplamente otimizadas para diversas arquiteturas, incluindo supercomputadores, clusters e redes heterogêneas. A padronização da interface garante a portabilidade do código, mesmo entre diferentes fabricantes e ambientes computacionais.

MPI oferece dois tipos principais de comunicação:

- Ponto a ponto (point-to-point): entre dois processos específicos.

- Coletiva (collective): envolvendo todos os processos de um grupo.

A seguir, apresentamos as principais funções da API MPI divididas em categorias, com exemplos práticos e discussões teóricas sobre seu uso.

## Operações Básicas do MPI

Estas operações são essenciais para qualquer programa paralelo usando MPI. Elas controlam o ambiente de execução e permitem que os processos se identifiquem e interajam.

### MPI_Init

Inicializa o ambiente MPI. Deve ser a primeira chamada em qualquer programa MPI.

```C 
MPI_Init(&argc, &argv);
```

### MPI_Finalize

Finaliza o ambiente MPI. Deve ser a última chamada no programa.

``` C
MPI_Finalize();
```


### MPI_Comm_size

Obtém o número total de processos em um comunicador.

```C
int size;
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

### MPI_Comm_rank

Retorna o identificador único do processo dentro de um comunicador.

```C
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

### MPI_Send e MPI_Recv

Operações ponto a ponto para envio e recebimento de mensagens.

```C
MPI_Send(&valor, 1, MPI_INT, destino, tag, MPI_COMM_WORLD);
MPI_Recv(&valor, 1, MPI_INT, origem, tag, MPI_COMM_WORLD, &status);
```

## Operações Coletivas

### MPI_Barrier

Sincroniza todos os processos.

```C
MPI_Barrier(MPI_COMM_WORLD);
```

### MPI_Bcast

Transmite dados do processo root para os demais.

```C
MPI_Bcast(&dados, 1, MPI_INT, root, MPI_COMM_WORLD);
```

### MPI_Gather

Coleta dados de todos os processos para o root.

```C
MPI_Gather(&local, 1, MPI_INT, buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### MPI_Scatter

Distribui dados do root para os demais.

```C
MPI_Scatter(buffer, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### MPI_Reduce

Aplica operação (como soma) e envia resultado ao root.

```C
MPI_Reduce(&local, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```

### MPI_Allreduce

Aplica operação e distribui o resultado a todos.

```C
MPI_Allreduce(&local, &resultado, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
```

## Operações de Controle

### MPI_Wtime

Retorna o tempo atual para medições.

```C
double inicio = MPI_Wtime();
double fim = MPI_Wtime();
```

### MPI_Status

Usado para obter metadados de mensagens recebidas.


### MPI_Initialized

Verifica se MPI já foi inicializado.

```C
int flag;
MPI_Initialized(&flag);
```

## Exemplos Finais Codificados em C

### Hello World

```C
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Olá do processo %d de %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
```

### Soma Vetorial com Reduce

```C
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local = rank + 1, soma;
    MPI_Reduce(&local, &soma, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Soma total: %d\n", soma);
    MPI_Finalize();
    return 0;
}
```

### Multiplicação de Matriz por Vetor

```C
#include <mpi.h>
#include <stdio.h>
#define N 4
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int A[N][N] = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12}, {13,14,15,16}};
    int x[N] = {1,1,1,1};
    int local_result = 0, result[N];

    for (int j = 0; j < N; j++)
        local_result += A[rank][j] * x[j];

    MPI_Gather(&local_result, 1, MPI_INT, result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultado: ");
        for (int i = 0; i < N; i++)
            printf("%d ", result[i]);
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
```

### Cálculo de Pi com MPI

```C
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#define PI25DT 3.141592653589793238462643
int main(int argc, char *argv[]) {
    int n = 100000, rank, size;
    double PI, h, sum = 0.0, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    h = 1.0 / (double)n;
    for (int i = rank + 1; i <= n; i += size) {
        x = h * ((double)i - 0.5);
        sum += 4.0 / (1.0 + x * x);
    }

    double local_pi = h * sum;
    MPI_Reduce(&local_pi, &PI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Pi aproximado: %.16f\nErro: %.16f\n", PI, fabs(PI - PI25DT));

    MPI_Finalize();
    return 0;
}
```

## Considerações Finais
Estas operações e exemplos cobrem o núcleo do desenvolvimento com MPI. Combinando essas operações, é possível projetar algoritmos paralelos escaláveis e portáteis.

### Exercícios

#### Questões de Teste

    1. Verdadeiro ou falso:
    (a) MPI é uma especificação de biblioteca de passagem de mensagens, não uma especificação de linguagem ou compilador.
    (b) No modelo MPI, os processos comunicam-se apenas por memória compartilhada.
    (c) MPI é útil para a implementação de paralelismo MIMD/SPMD.
    (d) Normalmente, um único programa MPI é escrito para rodar com um número geral de processos.
    (e) É necessário especificar explicitamente qual parte do código MPI será executada por processos específicos.

    2. Verdadeiro ou falso:
    (a) Um grupo e um contexto juntos formam um comunicador.
    (b) O comunicador padrão MPI_COMM_WORLD contém em seu grupo todos os processos iniciais e seu contexto é o padrão.
    (c) Um processo é identificado pelo seu rank no grupo associado a um comunicador.
    (d) O rank máximo é igual ao tamanho.

    3. Liste, na ordem requerida:
    (a) Funções MPI para controlar os procedimentos de início e término dos processos MPI.
    (b) Funções MPI para determinar o número de processos participantes e o identificador do processo atual.

    4. Suponha que um processo com rank 1 tenha iniciado a execução de MPI_SEND (buf, 5, MPI_INT, 4, 7, MPI_COMM_WORLD).
    (a) Qual processo deve iniciar o MPI_RECV correspondente para finalizar esta comunicação?
    (b) Escreva o MPI_RECV adequado.
    (c) O que será recebido?

    5. Nomeie as seguintes definições da semântica de comunicação do MPI:
    (a) Uma operação pode retornar antes de sua conclusão, e antes que o usuário possa reutilizar recursos (como buffers) especificados na chamada.
    (b) O retorno de uma chamada indica que os recursos podem ser reutilizados com segurança.
    (c) Uma chamada pode exigir a execução de uma operação em outro processo, ou comunicação com outro processo.
    (d) Todos os processos de um grupo precisam invocar o procedimento.

    6. Quando um processo chama MPI_RECV, ele espera pacientemente até que um envio correspondente seja postado. Se o envio correspondente nunca for postado, o receptor espera para sempre.
    (a) Nomeie essa situação.
    (b) Descreva uma solução para o problema.

    7. Forneça um segmento de programa funcionalmente equivalente usando envio não bloqueante para implementar a operação de envio bloqueante MPI_SEND.

    8. Nomeie as seguintes definições da semântica de comunicação do MPI:
    (a) Se um remetente posta duas mensagens para o mesmo receptor, e uma operação de recebimento corresponde a ambas, a mensagem postada primeiro será escolhida primeiro.
    (b) Não importa quanto tempo um envio tenha ficado pendente, ele sempre pode ser ultrapassado por uma mensagem enviada de outro processo.
    (c) A implementação do MPI garante por si só a justiça no envio?

    9. (a) Implemente uma operação de broadcast MPI one-to-all, onde um processo nomeado (root) envia os mesmos dados para todos os outros processos.
       (b) Qual(is) processo(s) deve(m) chamar essa operação?

    10. Suponha um array M × N de doubles armazenado no layout row-major em C na memória do sistema remetente.
    (a) Construa um tipo derivado contínuo MPI_newtype especificando uma coluna do array.
    (b) Escreva um MPI_Send para enviar a primeira coluna do array. Tente o mesmo para a segunda coluna. Note que o primeiro stride começa agora em array[0][1].

    11. Suponha quatro processos a, b, c, d, com oldrank no comm: 0, 1, 2, 3. Seja color = oldrank % 2 e key correspondente 7, 1, 0, 3. Identifique os novos grupos de newcomm, ordenados pelos newranks, após a execução de:
    MPI_COMM_SPLIT (comm, color, key, newcomm).

    12. Quais tipos de composição de programas paralelos são suportados por:
    (a) MPI_COMM_DUP (comm, newcomm) e por
    (b) MPI_COMM_SPLIT (comm, color, key, newcomm)?
    (c) As operações acima são exemplos de operações coletivas?

### Mini Projetos

    Implemente um programa MPI para um algoritmo de diferenças finitas 2D em um domínio quadrado com n × n = N pontos. Assuma stencil de 5 pontos (ponto atual e quatro vizinhos).
    Assuma pontos de fronteira fantasma para simplificar o cálculo nos pontos de borda (todos os stencils, incluindo os pontos de borda, são iguais).
    Compare os resultados obtidos, após um número especificado de iterações, em um único processo MPI e em um computador paralelo multicore, por exemplo, com até oito núcleos.
    Use modelos de desempenho para cálculo e comunicação para explicar seus resultados.
    Plote o tempo de execução em função do número de pontos N e em função do número de processos p, para, por exemplo, 10⁴ passos de tempo.
    
    P2. Use comunicação ponto a ponto do MPI para implementar as funções broadcast e reduce.
    Compare o desempenho da sua implementação com as operações globais MPI MPI_BCAST e MPI_REDUCE para diferentes tamanhos de dados e diferentes números de processos. 
    Use tamanhos de dados de até 10^4 doubles e até todos os processos disponíveis. 
    Plote e explique os resultados obtidos.

    P3. Implemente a soma de quatro vetores, cada um com N doubles, com um algoritmo similar ao algoritmo de redução. A soma final deve estar disponível em todos os processos. 
    Use quatro processos. 
    Cada um inicialmente gera seu próprio vetor. Use comunicação ponto a ponto MPI para implementar sua versão da soma dos vetores gerados. T
    este seu programa com vetores pequenos e grandes. 
    Comente os resultados e compare o desempenho da sua implementação com o MPI_ALLREDUCE. Explique quaisquer diferenças.