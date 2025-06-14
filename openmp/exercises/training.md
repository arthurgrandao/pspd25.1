# Treino

## Questão 1: Paralelização com processos e threads

Dada uma lista de números de entrada, produza a soma desses elementos por meio de um programa MPI/OMP, considerando o seguinte:

- A lista de números deve ser dividida uniformemente entre os processos MPI gerados em tempo de execução, incluindo o MASTER. Por exemplo, se a quantidade de números da lista é igual a 36 e o programa for instanciado com 3 processos, cada um deles deverá tratar um subgrupo de 12 números.
- Cada processo deve criar tantas threads quantas forem identificadas no arquivo de entrada. Por exemplo, se um processo tem um subgrupo com 12 números e a quantidade threads/processo for 2, cada thread do referido processo deve cuidar de um subgrupo menor de 6 números.

**Entrada**

Na primeira linha do arquivo de entrada está a quantidade de threads que devem ser instanciadas em cada processo MPI que for ativado. Na segunda linha do arquivo de entrada está a quantidade de números a serem trabalhados Na terceira linha do arquivo de entrada está a lista de números a serem somados.

**Saída**

O arquivo de saída contém o valor relativo à soma da lista de números do arquivo de entrada.

**Restrições**

- Na parte MPI do programa, usar apenas as primitivas MPI_Send, MPI_Recv;
- Na parte do OMP não é permitido usar o pragma de paralelismo com a opção reduction para consolidar resultados;
- Para uma determinada entrada, o programa deve produzir sempre a mesma saída, independente da quantidade de processos MPI instanciadas em tempo de execução;

Exemplo de Entrada 1
```
3
12
17 15 26 83 97 44 66 91 32 5 22 78
```
Exemplo de Saída 1
```
576
```

Exemplo de Entrada 2
```
4
8
126 18 77 82 6 21 11 20
```

Exemplo de Saída 2
```
361
```
Exemplo de Entrada 3
```
1
6
177 812 25 14 98 19
```
Exemplo de Saída 3
```
1145
```

**Possível solução:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#define MASTER 0

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, qtd_numbers, num_threads, * v, total = 0, total_temp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        scanf("%d", &num_threads);
        scanf("%d", &qtd_numbers);
        v = (int *) malloc(qtd_numbers * sizeof(int));
        for (int i = 0; i < qtd_numbers; i++)
            scanf("%d", &v[i]);

        int chunk = qtd_numbers/(size - 1);
        int index = chunk;
        for (int i = 1; i < size; i++) {
            if (i == size - 1) chunk += qtd_numbers % (size - 1);
            MPI_Send(&chunk, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num_threads, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&v[index * (i - 1)], chunk, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(&total_temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += total_temp;
        }
        printf("%d\n", total);
        free(v);
    } else {
        MPI_Recv(&qtd_numbers, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&num_threads, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        v = (int *) malloc(qtd_numbers * sizeof(int));
        MPI_Recv(v, qtd_numbers, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < qtd_numbers; i++) {
            #pragma omp atomic
            total += v[i];
        }
        MPI_Send(&total, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        free(v);
    }
    MPI_Finalize();
    return 0;
}
```
