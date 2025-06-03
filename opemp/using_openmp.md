# OpenMP

### Utilizando OpenMP para Escrever Programas Multithreaded

Um programa paralelo em um multiprocessador com memória compartilhada normalmente consiste em múltiplos threads. O sistema operacional gerencia esses threads, aplicando multitarefa se houver mais threads do que núcleos lógicos e realizando balanceamento de carga para manter todos os núcleos igualmente utilizados.

Embora programas multithreaded possam ser escritos usando bibliotecas de baixo nível como `pthreads` no UNIX, isso frequentemente resulta em código não portável e cheio de detalhes de baixo nível. O **OpenMP** é uma escolha melhor para escrever programas paralelos em sistemas de memória compartilhada.

OpenMP é uma **API**, não uma nova linguagem, projetada como um complemento para linguagens existentes como Fortran ou C/C++. Sua API consiste em:

- **Diretivas de compilador:** Instruções para o compilador gerar código paralelo. Em C/C++, são expressas como `#pragma`.
- **Funções de suporte:** Permitem que programadores explorem e controlem o paralelismo durante a execução.
- **Variáveis de ambiente (shell):** Permitem ajustar programas compilados para sistemas paralelos específicos.

#### Compilando e Executando um Programa OpenMP

Programas OpenMP são estruturados para permitir que um único thread inicial execute sequencialmente até que uma região paralela seja encontrada, momento em que um time de threads é formado. Cada thread executa o código especificado e, ao terminar, os threads recém-criados são encerrados, com o programa continuando com um único thread.

Uma **região paralela** é definida usando a diretiva `#pragma omp parallel`, seguida por um bloco estruturado de código. O thread que encontra essa diretiva torna-se o **mestre** do time formado. Há uma barreira implícita no final da região paralela: todos os threads devem terminar antes que o mestre retome a execução sequencial.

**"Hello World" com OpenMP**
```c
#include <stdio.h>
#include <omp.h>

int main() {
    printf ("Hello, world:");
    #pragma omp parallel
    printf (" %d", omp_get_thread_num()); // Executado por cada thread
    printf ("\n");
    return 0;
}
```
**Explicação:** O programa inicia com um único thread, que imprime "Hello, world:". Quando atinge `#pragma omp parallel`, um time de threads é criado. Cada thread imprime seu número exclusivo usando `omp_get_thread_num()`. Após todos os threads finalizarem, o programa continua com um único thread que imprime uma nova linha.

Para **compilar** um programa OpenMP usando GCC C/C++, usa-se a opção de linha de comando `-fopenmp`:

```
$ gcc -fopenmp -o hello-world hello-world.c
```

Para **executar** um programa OpenMP, o número de threads pode ser controlado com a variável de ambiente `OMP_NUM_THREADS`:

```
$ env OMP_NUM_THREADS=8 ./hello-world
```

- Se `OMP_NUM_THREADS` não for definida, o programa geralmente usará um número de threads igual ao de núcleos lógicos disponíveis.
- A ordem na qual os números dos threads são impressos pode variar a cada execução, devido ao agendamento pelo OpenMP e pelo sistema operacional.

**Controlando o Número de Threads:**

- **Variáveis de ambiente:**
  - `OMP_NUM_THREADS`: Define o número de threads para regiões paralelas.
  - `OMP_THREAD_LIMIT`: Limita o número máximo de threads que um programa pode usar, sobrepondo `OMP_NUM_THREADS`.

- **Funções OpenMP:**
  - `void omp_set_num_threads(int num_threads)`: Define o número de threads para regiões paralelas subsequentes.
  - `int omp_get_num_threads()`: Retorna o número de threads no time atual.
  - `int omp_get_max_threads()`: Retorna o número máximo de threads disponível.
  - `int omp_get_thread_num()`: Retorna o número exclusivo do thread dentro de seu time.

#### Monitorando um Programa OpenMP

Monitorar e medir o desempenho de um programa OpenMP é essencial para entender como ele roda em um sistema multi-core e determinar quantos núcleos ele realmente utiliza.

**Calculando Números de Fibonacci**
```c
#include <stdio.h>
#include <omp.h>

long fib(int n) { return (n < 2 ? 1 : fib(n - 1) + fib(n - 2)); }

int main() {
    int n = 45;
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        printf("%d: %ld\n", t, fib(n + t));
    }
    return 0;
}
```

**Explicação:** Este programa usa OpenMP para calcular e imprimir números de Fibonacci em paralelo. Cada thread calcula `fib(n + t)` onde `t` é seu número de thread.

**Medição do Tempo de Execução:**

Sistemas operacionais geralmente fornecem utilitários como `time` no Linux para medir o tempo de execução de programas. A saída normalmente inclui:

- **Tempo real (real):** Tempo total decorrido.
- **Tempo de usuário (user):** Tempo total de CPU gasto pelos núcleos executando o código do usuário.
- **Tempo de sistema (sys):** Tempo total de CPU gasto pelos núcleos executando chamadas do sistema.

Se a soma de `user` e `sys` for maior que `real`, indica que partes do programa rodaram simultaneamente em múltiplos núcleos lógicos.

**Monitores de Sistema:**

A maioria dos sistemas operacionais oferece monitores que mostram a carga computacional em núcleos individuais. Isso é informativo durante o desenvolvimento com OpenMP, mostrando como a carga nos núcleos lógicos diminui à medida que os threads terminam suas tarefas e como o sistema operacional pode migrar threads para balanceamento de carga.
