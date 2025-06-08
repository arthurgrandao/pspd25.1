# Questões de Provas Anteriores

## Questão 1

Elabore um programa MPI com N processos, sendo o master o responsável por inicializar o vetor e os slaves, responsáveis por imprimir uma porção do vetor, proporcional ao número de slaves identificados pelo programa.

**Possível Solução**:

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0
#define N 100

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == MASTER)
            printf("Necessário ter ao menos 2 processos\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == MASTER) {
        int * v = (int *) malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) v[i] = i;
        int chunk = N/(size - 1), remainder = N % (size - 1), offset = 0;

        for (int i = 1; i < size; i++) {
            int count = chunk + (i <= remainder ? 1 : 0);
            MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&v[offset], count, MPI_INT, i, 0, MPI_COMM_WORLD);
            offset += count;
        }
        free(v);
    } else {
        int chunk;
        MPI_Recv(&chunk, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int * recvbuff = (int *) malloc(chunk * sizeof(int));
        MPI_Recv(recvbuff, chunk, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d/%d (%d elements): ", rank, size, chunk);
        for (int i = 0; i < chunk; i++) printf("%d ", recvbuff[i]);
        printf("\n");
        free(recvbuff);
    }
    MPI_Finalize();
    return 0;
}
```

---

## Questão 2

O programa a seguir é uma implementação do cálculo da função Pi em modo serial, usando único espaço de endereçamento. Com base neste código crie uma versão MPI, de modo a permitir que os processos, em conjunto e colaborativamente, consigam encontrar o valor de Pi, pela distribuição do cálculo das áreas dos retângulos entre os processos.

```c
#include <stdio.h>

#define NUM_STEPS 8000000

int main(void) {
    double x, pi, sum = 0.0;
    double step;

    step = 1.0 / (double) NUM_STEPS;

    for (int i = 0; i < NUM_STEPS; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = sum * step;
    printf("Pi = %f\n", pi);

    return 0;
}
```
**Possível solução:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0
#define NUM_STEPS 8000000

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    double total_sum = 0.0, x, sum = 0.0, step = 1.0 / (double) NUM_STEPS;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int steps_per_process = NUM_STEPS/size;
    int start = rank * steps_per_process;
    int end = (rank == size - 1) ? NUM_STEPS : (rank + 1) * steps_per_process;

    for (int i = start; i < end; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    MPI_Reduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    
    if (rank == MASTER) {
        double pi = total_sum * step;
        printf("Pi = %.16f\n", pi);
    }
    MPI_Finalize();
    return 0;
}
```

---

## Questão 3

O objetivo do código a seguir é fazer a impressão do texto repetidas vezes, em função da constante MAX e da variável number, que é calculada a cada iteração do while. Crie uma versão MPI deste código com apenas três processos, de modo que a lógica original se mantenha, mas a impressão do texto ocorra de modo colaborativo (com divisão de trabalho mais igual possível) entre esses três processos.

```c
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX 1000

int main (void) {
    char texto_base[] = "abcdefghijklmnopqrstuvwxzyz 1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    /* a variavel indice aponta para o primeiro caracter do texto */
    int *indice = (int *) malloc (sizeof(int));
    *indice = 0;
    struct timeval tv;
    int number, tmp_index, i, cont = 0;
    while(cont < MAX) {
        gettimeofday(&tv, NULL);
        number = ((tv.tv_usec / 47) % 3) + 1;
        tmp_index = *indice;
        for (i = 0; i < number; i++) {
            if ( !( tmp_index + i > sizeof(texto_base) ) )
                fprintf(stderr, "%c", texto_base[tmp_index + i]);
        }
        *indice = tmp_index + i;
        if (tmp_index + i > sizeof(texto_base)) {
            fprintf(stderr, "\n");
            *indice = 0;
        } /* fim-if */
        cont++;
    } /* fim-while */

    printf("\n");
    return 0;
} /* fim-main */
```
---

## Questão 4

O código MPI a seguir faz a impressão colaborativa de um vetor dinâmico, no qual (i) o MASTER faz a inicialização do vetor e (ii) cada processo faz a impressão de uma quantidade de elementos desse vetor, de acordo com o número de processos ativado. Percebe-se, no entanto, que esse programa possui um problema de alto consumo de memória, uma vez que todos os processos são obrigados a alocar o vetor completo (linha 15), embora façam a impressão de apenas um subconjunto desse vetor (para um vetor de 1 milhão de inteiros e 5 processos, serão alocados 5 milhões de inteiros, gerando alocação excessiva de memória). Promova as alterações nesse código, de modo a reduzir o consumo de memória e, ao mesmo tempo, garantir a impressão equitativa do vetor entre os processos.

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char* argv[]) {
    int rank, nprocs, *v, tamvet;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == MASTER) {
        printf("Tamanho do vetor: ");
        fflush(stdout);
        scanf("%d", &tamvet);
    } /* fim-if */

    MPI_Bcast(&tamvet, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    v = (int *) malloc(tamvet * sizeof(int));

    if (rank == MASTER) {
        for (int i = 0; i < tamvet; i++) {
            v[i] = (i + 1) * 10;
        }
    } /* fim-if */

    MPI_Bcast(v, tamvet, MPI_INT, MASTER, MPI_COMM_WORLD);

    int chunk = tamvet / nprocs;
    int ini = rank * chunk;

    if (rank == (nprocs - 1)) {
        chunk = chunk + (tamvet % nprocs);
    }

    int fim = ini + chunk;

    printf("%d/%d: ", rank, nprocs);
    for (int i = ini; i < fim; i++) {
        printf("%d ", v[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
} /* fim-main */
```

**Possível solução:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char* argv[]) {
    int rank, nprocs, *v, tamvet;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == MASTER) {
        printf("Tamanho do vetor: ");
        fflush(stdout);
        scanf("%d", &tamvet);
        v = (int *) malloc(tamvet * sizeof(int));
        for (int i = 0; i < tamvet; i++) v[i] = (i + 1) * 10;

        int chunk = tamvet/nprocs;
        int remainder = tamvet % nprocs;
        int offset = 0;
        for (int i = 0; i < nprocs; i++) {
            int base = chunk + (i < remainder ? 1 : 0);
            MPI_Send(&base, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&v[offset], base, MPI_INT, i, 0, MPI_COMM_WORLD);
            offset += base;
        }
        free(v);
    }

    MPI_Recv(&tamvet, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    v = (int *) malloc(tamvet * sizeof(int));
    MPI_Recv(v, tamvet, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("%d/%d (tamanho %d): ", rank, nprocs, tamvet);
    for (int i = 0; i < tamvet; i++) printf("%d ", v[i]);
    printf("\n");
    free(v);
    MPI_Finalize();
    return 0;
}
```

---

## Questão 5

Utilizando a biblioteca MPI, elabore um programa multi-processos para somar os elementos de duas matrizes A e B, quadradas (int ou float), para gerar a matriz C, seguindo as seguintes regras:

- O programa deve conter um processo master e quatro processos workers que deverão trabalhar em conjunto para garantir a realização de soma dos elementos das matrizes A e B;
- Supor que as matrizes são de 16 posições e as matrizes A e B devem ser inicializadas com números randômicos
- As operações de soma devem ser distribuídas uniformemente entre os workers;
- Ao final, a matriz C resultante deve ser impressa (em colunas, formato de matriz) pelo processo master

**Possível solução:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MASTER 0
#define N 4

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, a_line[N], b_line[N], c_line[N];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        int A[N][N], B[N][N], C[N][N];
        printf("Master: iniciando as matrizes\n");
        printf("Matriz A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                printf("%d ", A[i][j]);
            }
            printf("\n");
        }
        
        printf("Matriz B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[i][j] = rand() % 10;
                printf("%d ", B[i][j]);
            }
            printf("\n");
        }

        for (int i = 1; i < size; i++) {
            MPI_Send(A[i - 1], N, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(B[i - 1], N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < size; i++) MPI_Recv(C[i - 1], N, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Matriz C (resultado):\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) printf("%d ", C[i][j]);
            printf("\n");
        }
    } else {
        MPI_Recv(a_line, N, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_line, N, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; i++) c_line[i] = a_line[i] + b_line[i];
        MPI_Send(c_line, N, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
```

---

## Questão 6

Elabore um programa MPI que imprima um vetor de 100 posições (de tipo int), considerando o seguinte:

- O vetor deve ser impresso da posição 0 até a posição 99, nesta ordem;
- O master deve inicializar o vetor de 100 posições da seguinte forma: `v[i] = i`;
- O master deve distribuir a impressão entre os workers de modo que todos possam imprimir pelo menos uma porção do vetor;
- Cada worker, uma vez acionado, deve imprimir o vetor a partir do ponto de impressão recebido do master;
- Considerar que este programa pode ser executado por, no máximo, 6 processos (1 master e 5 workers);
- O número de posições a serem impressas pelo worker deve obedecer a um offset dinâmico, ou seja, um valor randômico – menor que 15 – que é calculado por cada processo, no momento em que é acionado para imprimir o vetor;
- O programa deve controlar a impressão de modo que o vetor inteiro seja impresso, mas nenhuma posição seja impressa mais de uma vez. Por exemplo, se o worker anterior imprimiu até a posição 18 e o offset dinâmico calculado foi 10, a thread atual deve imprimir da posição 19 considerando 10 posições adiante;
- A ação dos workers e do master acaba quando o vetor de 100 posições tiver sido todo impresso;

**Possível solução:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#define MASTER 0
#define N 100

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, *v, elements = 0, chunk;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        int proc = 1;
        v = (int *) malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) v[i] = i;

        while (elements < N) {
            chunk = rand() % 15 + 1;
            if (elements + chunk > N) chunk = N - elements;
            MPI_Send(&chunk, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
            MPI_Send(&v[elements], chunk, MPI_INT, proc, 0, MPI_COMM_WORLD);
            elements += chunk;
            proc++;
            if (proc >= size) proc = 1;
        }
        int end_signal = 0;
        for (int i = 1; i < size; i++)
            MPI_Send(&end_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    } else {
        while (true) {
            MPI_Recv(&chunk, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (chunk == 0) break;
            v = (int *) malloc(chunk * sizeof(int));
            MPI_Recv(v, chunk, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d/%d (size %d): ", rank, size, chunk);
            for (int i = 0; i < chunk; i++) printf("%d ", v[i]);
            printf("\n");
            free(v);
        }
    }
    MPI_Finalize();
    return 0;
}
```

---

## Questão 7

Observe o seguinte código MPI, cujo objetivo é conseguir gravar 100 elementos do vetor data em arquivo:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define FILE_NAME "file.bin"
#define MAX 100

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int data[MAX];
    MPI_File fh;

    int chunk = MAX / size;
    int start = rank * chunk * sizeof(int);
    if (rank == (size - 1)) chunk += (MAX % size);
    for (int i = 0; i < chunk; i++)
        data[i] = rank * chunk + i + 1;

    MPI_File_open(MPI_COMM_WORLD, FILE_NAME, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_write_at(&fh, start, data, chunk, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Finalize();
    /* fim-main */
}
```

**Afirmações:**

Considerando o propósito definido para o código, avalie as afirmativas e, a seguir, marque a opção correta:
I - O código não funciona adequadamente porque a função da linha 21 necessita um laço para garantir que cada processo faça a escritados elementos sob sua responsabilidade na posição correta do arquivo.

II - Este código funciona adequadamente e a instrução da linha 17 garante que os valores sequenciais, de 1 a 100, no vetor data, independente do número de processos.

III - O código apresentado não funciona adequadamente porque a função de escrita (linha 21) exige que o vetor a ser gravado seja dividido em partes iguais entre os processos MPI.

**Alternativas:**

a. Nenhuma das alternativas está correta

b. Apenas a afirmativa III está correta

c. Apenas a afirmativa II está correta

d. Apenas a afirmativa I está correta

e. Apenas as afirmativas I e III estão corretas

**Resposta Correta:**

Nenhuma das alternativas está correta


**Explicação:**: 

O código funciona bem da forma como está (afirmativa I é falsa). A afirmativa II é falsa porque não há gravação de todos os valores de 1 a 100 (no entanto, o comando equilibra a gravação entre o número de processos). Afirmativa III é falsa (a linha 17 garante que eventuais sobras do vetor sejam gravadas pelo último processo com a função MPI_File_write_at).
