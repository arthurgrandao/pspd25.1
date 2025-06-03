# OpenMP

### Executando Tarefas Independentes em Paralelo

O OpenMP oferece uma alternativa à paralelização de loops, especialmente quando o número e tamanho de tarefas individuais não são conhecidos antecipadamente ou quando o algoritmo do problema requer criação dinâmica de tarefas. As tarefas do OpenMP permitem que porções de um programa sejam executadas concorrentemente como unidades independentes de trabalho.

**Exemplo de Soma de Inteiros**
Este exemplo demonstra o uso de um número fixo de tarefas para somar inteiros de 1 até `max`.

```c
#include <stdio.h>
#include <omp.h>

int main (int argc , char *argv[]) {
    int max; sscanf (argv[1], "%d", &max);
    int tasks; sscanf (argv[2], "%d", & tasks);
    if (max % tasks != 0) return 1;
    int sum = 0;
    #pragma omp parallel
    {
        #pragma omp single
        for (int t = 0; t < tasks; t++) {
            #pragma omp task
            {
                int local_sum = 0;
                int lo = (max / tasks) * (t + 0) + 1;
                int hi = (max / tasks) * (t + 1) + 0;
                // printf ("%d: %d..%d\n", omp_get_thread_num(), lo, hi);
                for (int i = lo; i <= hi; i++)
                    local_sum = local_sum + i;
                #pragma omp atomic
                sum = sum + local_sum;
            }
        }
    }
    printf ("%d\n", sum);
    return 0;
}
```
- A execução paralela geral é definida por um bloco `#pragma omp parallel`.
- Dentro deste bloco paralelo, a diretiva `#pragma omp single` é usada. Isso garante que o loop `for` responsável por criar as tarefas seja executado por **exatamente uma thread** na equipe (não necessariamente a thread mestre), prevenindo criação redundante de tarefas. Outras threads esperam em uma barreira implícita no final da diretiva `single`, a menos que `nowait` seja especificado.
- Cada tarefa é definida por `#pragma omp task`. Neste exemplo, cada tarefa computa a soma de inteiros dentro de um subintervalo mutuamente disjunto.
- Para combinar os resultados dessas tarefas independentes, uma diretiva `#pragma omp atomic` é usada ao adicionar a `soma_local` à variável `soma`. Isso garante acesso exclusivo à variável compartilhada `soma`, prevenindo condições de corrida.
- Quando uma nova tarefa é criada, a tarefa criadora continua sua execução sem atraso; a nova tarefa tem seu próprio ciclo de vida independente. No entanto, todas as tarefas criadas dentro de uma região paralela devem completar antes que a própria região paralela possa terminar devido a uma barreira implícita.

**Exemplo de Números de Fibonacci**

```c
#include <stdio.h>
#include <omp.h>

long fib (int n) { return (n < 2 ? 1 : fib (n - 1) + fib (n - 2)); }

int main (int argc , char *argv[]) {
    int max; sscanf (argv[1], "%d", &max);
    #pragma omp parallel
    #pragma omp single
    for (int n = 1; n <= max; n++)
        #pragma omp task
        printf ("%d: %d %ld\n", omp_get_thread_num(), n, fib (n));
    return 0;
}
```
Este exemplo ilustra como computar números de Fibonacci usando tarefas, destacando a natureza dinâmica das tarefas.

- Similar ao exemplo de soma, uma única thread dentro de uma região paralela é usada para iniciar todas as tarefas.

### Diretiva de Tarefa OpenMP (`#pragma omp task`)

Cria uma nova tarefa que executa um bloco estruturado.

- A tarefa pode ser executada imediatamente ou adiada para execução posterior por qualquer thread na equipe.
- **Cláusula `final(expressao-logica-escalar)`**: Se a expressão avalia como `true`, a tarefa criada não gerará nenhuma nova subtarefa. Em vez disso, qualquer código que teria gerado novas subtarefas é executado dentro da tarefa atual. Isso é útil para evitar overhead quando tarefas se tornam muito pequenas (ex.: abaixo de um certo limite).
- **Cláusula `if([task:]expressao-logica-escalar)`**: Se a expressão avalia como `false`, uma tarefa não adiada é criada, significando que a tarefa criadora é suspensa até que a tarefa criada termine.

**Exemplo de Quicksort**
O Quicksort é apresentado como um exemplo principal onde tarefas são benéficas porque o número e tamanho de subproblemas (tarefas) não podem ser conhecidos antecipadamente.

```c
void par_qsort (char ** data, int lo, int hi, int (*compare) (const char *, const char*)) {
    if (lo > hi) return;
    int l = lo;
    int h = hi;
    char *p = data[(hi + lo) / 2];
    while (l <= h) {
        while (compare(data[l], p) < 0) l++;
        while (compare(data[h], p) > 0) h--;
        if (l <= h) {
            char *tmp = data[l]; data[l] = data[h]; data[h] = tmp;
            l++; h--;
        }
    }
    #pragma omp task final(h - lo < 1000)
    par_qsort (data, lo, h, compare);
    #pragma omp task final(hi - l < 1000)
    par_qsort (data, l, hi, compare);
}
```
- As chamadas recursivas no algoritmo Quicksort são convertidas em novas tarefas, permitindo que sejam executadas independentemente e potencialmente em paralelo.
- A cláusula `final` (ex.: `final(h - lo < 1000)`) é usada para definir um limite. Se a parte do array a ser ordenada é menor que este limite (ex.: 1000 elementos), nenhuma nova tarefa é criada, e a ordenação é executada sequencialmente dentro da tarefa atual. Isso ajuda a reduzir o overhead associado à criação de tarefas.
- A função `par_qsort` deve ser chamada dentro de uma região paralela por exatamente uma thread (usando `#pragma omp parallel` e `#pragma omp single`).
- Embora o Quicksort paralelo possa ser mais rápido, o speedup pode não ser proporcional ao número de threads devido a partes sequenciais do algoritmo (como a fase de particionamento) e à Lei de Amdahl.

## Combinando os Resultados de Tarefas Paralelas

Quando tarefas não são inteiramente independentes e seus resultados precisam ser combinados, diretivas específicas do OpenMP são necessárias para garantir agregação correta.

**Exemplo: Quicksort Revisitado**
Esta versão do Quicksort conta o número de trocas de elementos durante as fases de partição.

```c
int par_qsort ( char ** data , int lo , int hi ,
int (*compare)(const char *, const char*)) {
    if (lo > hi) return 0;
    int l = lo;
    int h = hi;
    char *p = data[(hi + lo) / 2];
    int count = 0;
    while (l <= h) {
        while (compare( data[l], p) < 0) l++;
        while (compare( data[h], p) > 0) h--;
        if (l <= h) {
            count++;
            char *tmp = data[l]; data[l] = data[h]; data[h] = tmp;
            l++; h--;
        }
    }

    int locount, hicount;
    #pragma omp task shared(locount) final(h - lo < 1000)
    locount = par_qsort (data, lo, h, compare);
    #pragma omp task shared(hicount) final(hi - l < 1000)
    hicount = par_qsort (data, l, hi, compare);
    #pragma omp taskwait
    return count + locount + hicount;
}
```

- Os resultados das chamadas de tarefa recursivas (número de trocas em sub-arrays) são armazenados em `locount` e `hicount`.
- A **cláusula `shared`** (ex.: `shared(locount)`) é essencial para `locount` e `hicount` porque essas variáveis recebem valores dentro das tarefas recém-criadas, e seus valores devem ser acessíveis à tarefa criadora que irá combiná-los.
- A **diretiva `#pragma omp taskwait`** é crucial aqui. Ela serve como uma **barreira explícita**, forçando a tarefa que a encontra a esperar até que todas as suas tarefas filhas diretamente geradas (subtarefas) tenham completado sua execução. Isso previne que a tarefa criadora tente somar `locount` e `hicount` antes que tenham sido adequadamente definidos por suas respectivas tarefas.
- É importante diferenciar `taskwait` da barreira implícita no final de uma região paralela: `taskwait` especificamente sincroniza com as próprias subtarefas de uma tarefa, enquanto a barreira implícita no final de uma seção paralela garante que todas as tarefas criadas dentro de toda aquela região paralela tenham terminado.

## Principais Conceitos de Tarefas OpenMP

### Características das Tarefas

1. **Criação Dinâmica**: Tarefas podem ser criadas dinamicamente durante a execução, ao contrário dos loops paralelos que têm estrutura estática.

2. **Execução Independente**: Cada tarefa tem seu próprio ciclo de vida e pode ser executada por qualquer thread disponível na equipe.

3. **Flexibilidade**: Adequadas para algoritmos com paralelismo irregular ou recursivo.

### Sincronização de Tarefas

- **Barreira Implícita**: No final de uma região paralela, todas as tarefas devem completar.
- **`taskwait`**: Espera apenas pelas subtarefas diretas da tarefa atual.
- **`taskgroup`**: Permite sincronização mais refinada de grupos de tarefas.

### Otimizações

- **Cláusula `final`**: Evita overhead de criação de tarefas muito pequenas.
- **Cláusula `if`**: Controla quando tarefas devem ser criadas ou executadas diretamente.
- **Balanceamento de Carga**: O runtime do OpenMP distribui tarefas automaticamente entre threads disponíveis.

### Casos de Uso Ideais

- Algoritmos recursivos (Quicksort, busca em árvores)
- Processamento de listas ligadas
- Algoritmos com paralelismo irregular
- Situações onde o número de unidades de trabalho não é conhecido antecipadamente

As tarefas OpenMP fornecem uma abstração poderosa para paralelização que complementa os loops paralelos, oferecendo maior flexibilidade para algoritmos complexos e estruturas de dados dinâmicas.