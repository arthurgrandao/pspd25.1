# Questões de Provas Anteriores

## Questão 1

O código a seguir foi compilado com OpenMP e o binário tem o nome t1. Com base no código fonte, analise as afirmações e marque V para as verdadeiras e F para as falsas.

```c
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    int i = 0;
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 1)
            i = i + 10;
    }
    printf("i = %d\n", i);
    return 0;
}
```

**Afirmações:**

I - A execução com o comando `OMP_NUM_THREADS=4 t1` vai imprimir o valor 40.

II - Se a linha 9 for suprimida, o binário equivalente acionado com o comando `OMP_NUM_THREADS=3 t1` imprimirá sempre o valor 30.

III - Se na linha 7 for acrescentada a declaração `private(i)` e houver supressão da linha 9, o binário equivalente acionado com o comando `OMP_NUM_THREADS=6 t1`, o programa vai imprimir 60.

**Alternativas:**

a. Apenas a afirmativa I está correta

b. Apenas as afirmativas I e II estão corretas

c. Apenas as afirmativas I e III estão corretas

d. Nenhuma das opções apresentadas corresponde às afirmativas apresentadas

e. Apenas as afirmativas II e III estão corretas

**Resposta correta:**

Nenhuma das opções apresentadas corresponde às afirmativas apresentadas

---

## Questão 2

Analise o código a seguir e responda o que se segue

```c
#include <stdio.h>
#include <omp.h>

int main() {
    int tid = 0, nthreads = 0;
    printf("\nRegião serial (thread única)\n\n");
    #pragma omp parallel
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("Região paralela (thread %d de %d threads)\n", tid, nthreads);
    } /* fim-pragma */
    printf("\nRegião serial (thread única)\n\n");
    #pragma omp parallel num_threads(4)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("Região paralela (thread %d de %d threads)\n", tid, nthreads);
    } /* fim-pragma */
    printf("\nRegião serial (thread única)\n\n");
    return 0;
} /* fim-main */
```
**Afirmações:**
1. Se `OMP_NUM_THREADS=6`, na segunda região paralela desse código (linhas 13 a 18), serão geradas 10 threads e, portanto, 10 impressões (linha 17).
2. Se a linha 15 for movida para ficar fora da região paralela (entre as linhas 11 e 13), esse código passa a ser não compilável, pois não é possível saber o número de threads em uma região serial do código.
3. Esse código é mais apropriado para funcionar em arquiteturas UMA (Uniform Memory Access) ou de memória compartilhada do que em arquiteturas NUMA (Non Uniform Memory Access).

**Resposta correta:**

1 - Errada

2 - Errada

3 - Certa

---

## Questão 3

O código a seguir foi compilado com OpenMP e o binário tem o nome t1. Com base no código fonte, analise as afirmações e marque V para as verdadeiras e F para as falsas.

```c
#include <stdio.h>
#include <omp.h>
#include <string.h>
#define MAX 100

int main(int argc, char *argv[]) {
    #pragma omp parallel
    {
        int soma = 0;
        #pragma omp for
        for (int i = 0; i < MAX; i++) {
            soma += omp_get_num_threads() * i;
        } /* fim-for */
        printf("Thread[%d] iterou %d vezes\n", 
               omp_get_thread_num(), soma);
    } /* fim omp parallel */
    return 0;
}
```

**Afirmações:**

I - A execução com o comando `OMP_NUM_THREADS=4 t1` vai imprimir que cada thread foi executada 25 vezes.

II - Se este programa for acionado tendo a variável `OMP_NUM_THREADS` um valor maior do que o número de núcleos da máquina, apenas as threads equivalentes ao número de núcleos serão criadas

III - Se o programa for executado numa máquina com 10 núcleos de processamento e a variável `OMP_NUM_THREADS` estiver com valor igual a 20, o programa não será ativado

**Resposta Correta:**

1 - Errada

2 - Errada

3 - Errada

---

## Questão 4

Analise o programa a seguir:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#define MAX 14

int main(int argc, char* argv[]){
    int sum=0;

    #pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (int i=1; i<=MAX; i++) {
        printf("%4d @ %d\n", i, omp_get_thread_num());
        sleep(i<4? i+1:1);
        sum+=i;
    } /* fim-for */
    printf("Soma = %d\n", sum);
    return 0;
} /*fim-main*/
```
**Alternativas:**

I - O escalonamento com static,1 produz um desempenho pior do que escalonamento com dynamic,1

II - O escalonamento com static,2 produz um desempenho melhor do que o escalonamento com static,1

III - O escalonamento com static,3 tem desempenho pior do que o escalonamento com dynamic,2

**Resposta Correta:**

I - Certo

II - Errado

III - Certo


---

## Questão 5

O código a seguir foi compilado com OpenMP e o binário tem o nome t1. Com base no código fonte, analise as afirmações e marque V para as verdadeiras e F para as falsas.

```c
#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("%d\n", omp_get_num_threads());
    #pragma omp parallel
    {
    }
    printf("%d\n", omp_get_max_threads());
    return 0;
}
```
**Afirmações:**

I - A execução com o comando `OMP_NUM_THREADS=4 t1` vai imprimir o valor 4 na linha 5 e o valor 12 na linha 10, se o computador onde esse programa estiver rodando tiver 12 núcleos

II - Este programa vai imprimir sempre o valor 1 na linha 5, independente do número de threads definidas na variável `OMP_NUM_THREADS`

III - O comando da linha 10 vai imprimir sempre o valor 1, uma vez que este está fora da região paralela definida pelo pragma omp parallel

**Alternativas:**

a. Nenhuma das opções consegue julgar as afirmativas apresentadas

b. Apenas as afirmativas I e II estão corretas

c. Apenas as afirmativas I e III estão corretas

d. Apenas as afirmativas II e III estão corretas

e. Apenas a afirmativa I está correta

**Resposta Correta:**

Nenhuma das opções consegue julgar as afirmativas apresentadas

---

## Questão 6

Sabe-se que #pragma omp parallel for faz a divisão de iterações do laço de execução entre as threads instanciadas no momento da execução. Elabore um programa que seja capaz de manipular um vetor de entrada com tamanho aleatório e que possa identificar/imprimir o número de índices do vetor de entrada que cada thread do programa ficou responsável (número de iterações por thread) com o pragma citado.

**Resposta:**

```c
#include <stdio.h>
#include <omp.h>
#include <string.h>
#define MAX 10

int main(int argc, char *argv[]) {
    int vetor[omp_get_max_threads()];
    memset(&vetor, 0, 4 * omp_get_max_threads());

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < MAX; i++) {
            vetor[omp_get_thread_num()]++;
        }
    }

    for (int i = 0; i < omp_get_max_threads(); i++) {
        printf("Thread[%d] iterou %d vezes \n", i, vetor[i]);
    }

    return 0;
}
```

**Resposta Alternativa:**
```c
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

typedef struct indices {
  int inicio;
  int fim;
} indices;

int main(void) {
  srand(42);
  int tam = rand() % 1000 + 1;
  int vetor[tam];
  memset(&vetor, 0, 4*tam);

  int num_threads = omp_get_max_threads();
  
  indices *ind = calloc(num_threads, sizeof(indices));

  printf("Número de threads: %d\n", num_threads);
  printf("Tamanho do vetor: %d\n", tam);

  #pragma omp parallel for 
  for (int i = 0; i < tam; i++) {
    int tid = omp_get_thread_num();
    if (ind[tid].inicio == 0 && ind[tid].fim == 0 && tid) {
      ind[tid].inicio = i;
      ind[tid].fim = i;
    } else {
      if (i < ind[tid].inicio) ind[tid].inicio = i;
      if (i > ind[tid].fim) ind[tid].fim = i;
    }

    vetor[i] = omp_get_thread_num(); // manipula o vetor 
  }
  
  // Extrai o resultado
  for (int i = 0; i < num_threads; i++) {
    printf("Thread %d: inicio = %d, fim = %d\n", i, ind[i].inicio, ind[i].fim);
  }
  return 0;
}
```

---

## Questão 7

No código a seguir, os pragmas declarados nas linhas 11 e 14 garantem a divisão equilibrada do trabalho entre o total de threads especificadas na variável de ambiente `OMP_NUM_THREADS`. Apresente uma nova versão desse código que garanta a distribuição equilibrada de trabalho entre as threads (de acordo com o valor de `OMP_NUM_THREADS`), considerando apenas o pragma de paralelização descrito na linha 11 (ou seja, assuma a não existência do pragma da linha 14).

```c
#include <stdio.h>
#include <omp.h>
#define TAM 12

int main() {
	int A[TAM], B[TAM], C[TAM];
	int i;
	for (i = 0; i < TAM; i++) {
		A[i] = 2*i - 1;
		B[i] = i + 2;
	}
	#pragma omp parallel
	{
		int tid = ome_get_thread_num();
		#pragma omp for
		for (i = 0; i < TAM; i++) {
			C[i] = A[i] + B[i];
			printf("Thread[%d] calculou C[%d]\n", thid, i);
		}
	}
	for (i = 0; i < TAM; i++)
		printf("C[%d] = %d\n", i, C[i]);
}
```

**Possível Resposta:**

```c
#include <stdio.h>
#include <omp.h>
#define TAM 12

int main() {
	int A[TAM], B[TAM], C[TAM];
	int i;
	for (i = 0; i < TAM; i++) {
		A[i] = 2*i - 1;
		B[i] = i + 2;
	}
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk = TAM / num_threads;
        int start = tid * chunk;
        int end = (tid == num_threads - 1) ? TAM : start + chunk;

		for (int i = start; i < end; i++) {
			C[i] = A[i] + B[i];
			printf("Thread[%d] calculou C[%d]\n", tid, i);
		}
	}
	for (i = 0; i < TAM; i++)
		printf("C[%d] = %d\n", i, C[i]);
}
```

---

## Questão 8

Elabore um programa OpenMP que imprima os elementos de uma matriz M [100:8] posições (do tipo int), considerando o seguinte:
- Apenas uma das threads deve inicializar a matriz, com a fórmula: M[i][j]=i+j
- A matriz deve ser impressa de modo colaborativo por todas as threads ativadas, da linha 0 até a linha 99, nesta ordem
- Cada thread deve imprimir pelo menos uma das linhas da matriz
- Cada thread, uma vez acionada, deve imprimir a matriz a partir da linha que ainda não foi impressa
- Considerar que este programa pode ser executado por, no máximo, 6 threads
- O número de posições a serem impressas deve obedecer a um offset dinâmico, ou seja, um valor randômico – menor que 15 – que é calculado por cada thread, no momento em que a thread é escalonada para imprimir a matriz
- O programa deve controlar a impressão de modo que a matriz inteira seja impressa em ordem e de modo que nenhuma posição seja impressa mais de uma vez. Por exemplo, se a matriz já foi impressa até a linha 18 e o offset dinâmico calculado foi 10, a thread atual deve imprimir da linha 19 considerando 10 posições adiante
- O programa termina quando a matriz M inteira, linha por linha, tiver sido impressa.

**Possível Resposta:**

```c
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define LINHAS 100
#define COLUNAS 8

int main(void) {
    srand(42); 
    int M[LINHAS][COLUNAS];
    int linhas_impressas = 0;

    #pragma omp parallel shared(M, linhas_impressas)
    {
        #pragma omp single
        {
            for (int i = 0; i < LINHAS; i++) {
                for (int j = 0; j < COLUNAS; j++) {
                    M[i][j] = i + j;
                }
            }
        }

        #pragma omp barrier

        int tid = omp_get_thread_num();
        
        while (linhas_impressas < LINHAS) {
            int offset = rand() % 15;
            !offset ? offset = 1 : offset;

            #pragma omp critical
            {
                if (linhas_impressas + offset > LINHAS) {
                    offset = LINHAS;
                }

                for (int i = linhas_impressas; i < linhas_impressas + offset && i < LINHAS; i++) {
                    printf("Thread %d imprimindo M[%d]: ", tid, i);
                    for (int j = 0; j < COLUNAS; j++) {
                        printf("%d ", M[i][j]);
                    }
                    printf("\n");
                }
                linhas_impressas += offset;
            }
        }
    }

    return 0;
}
```

---

## Questão 9

Fazer o programa imprimir um vetor de 100 posições em ordem. Inicie o vetor com v[i] = i. Utilizar offset dinâmico para cada thread. Exemplo: a thread anterior fez a última impressão no item 18, a thread tem um offset dinâmico de 3. Então a thread atual deve imprimir os itens 19, 20, 21.

**Possível Resposta:**

```c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 100

int main() {
    srand(42); 
    int v[SIZE];
    int num_impressos = 0;
    int offset;

    for (int i = 0; i < SIZE; i++) {
        v[i] = i;
    }

    #pragma omp parallel
    {
        while (num_impressos < SIZE) {
            offset = rand() % 15;
            if (offset == 0) offset = 1;

            #pragma omp critical
            {
                if (num_impressos < SIZE) {
                    int tid = omp_get_thread_num();
                    int limite = num_impressos + offset;
                    if (limite > SIZE) limite = SIZE;

                    printf("\nTID %d, %d->%d:\n", tid, num_impressos, limite);
                    for (int i = num_impressos; i < limite; i++) {
                        printf("%d ", v[i]);
                    }
                    num_impressos = limite;
                }
            }
        }
    }
    return 0;
}
```
