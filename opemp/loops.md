# OpenMP

### Paralelizando Loops com Iterações Independentes

Muitos programas intensivos de CPU passam a maior parte do tempo em loops, tornando-os candidatos ideais para paralelização. O OpenMP fornece a **diretiva `#pragma omp parallel for`** como uma maneira conveniente de paralelizar tais loops. Quando esta diretiva é aplicada, as iterações do loop são divididas entre e executadas por múltiplas threads em uma equipe. A thread que encontra a diretiva torna-se a thread mestre, e outras threads "escravas" são criadas para formar uma equipe. Uma vez que todas as iterações são executadas, uma barreira implícita no final do loop `omp parallel for` garante que todas as threads se sincronizem e as threads escravas sejam terminadas, com a thread mestre retomando a execução sequencial.

Para que um loop `for` seja paralelizado com `omp for` (ou `omp parallel for`), ele deve estar em **forma canônica**. Isso significa:

- A variável do loop é privada para cada thread e deve ser um inteiro (com ou sem sinal) ou um ponteiro.
- A variável do loop não deve ser modificada dentro de qualquer iteração.
- A condição do loop deve ser uma expressão relacional simples.
- O incremento deve especificar uma mudança por uma expressão aditiva constante.
- O número de iterações deve ser conhecido antes do loop começar.

**Compartilhamento de dados** entre threads em uma região paralela pode ser especificado usando cláusulas:

- **`shared(lista)`**: Variáveis na lista são compartilhadas, significando que todas as threads acessam a mesma cópia.
- **`private(lista)`**: Cada thread obtém sua própria cópia local das variáveis na lista.
- **`firstprivate(lista)`**: Similar ao `private`, mas cada variável é inicializada com o valor que tinha antes da região paralela.
- **`lastprivate(lista)`**: Similar ao `private`, mas a variável original é atualizada com o valor final de uma das cópias privadas quando a região paralela termina.
- Por padrão, variáveis automáticas declaradas fora de um construto paralelo são compartilhadas, enquanto aquelas declaradas dentro são privadas; variáveis estáticas e alocadas dinamicamente são compartilhadas.

**Imprimindo Inteiros**
```c
#include <stdio.h>
#include <omp.h>

int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    #pragma omp parallel for // Iterações do loop for são divididas entre threads
    for (int i = 1; i <= max; i++)
        printf ("%d: %d\n", omp_get_thread_num (), i); // Cada thread imprime seu ID e um número de iteração
    return 0;
}
```

**Explicação**: A diretiva `omp parallel for` faz com que as iterações do loop sejam distribuídas. Cada thread `i` imprimirá seu número de thread e o valor `i` da iteração que está processando. A variável `max` pode ser compartilhada pois é apenas lida dentro do loop, enquanto `i` deve ser privada para a iteração de cada thread.

### Loops Aninhados

Para **loops aninhados**, o OpenMP oferece múltiplas abordagens de paralelização:

#### 1. Paralelizando apenas o loop mais externo:

```c
#pragma omp parallel for
for (int i = 1; i <= max; i++)
    for (int j = 1; j <= max; j++)
        printf ("%d: (%d,%d)\n", omp_get_thread_num (), i, j);
```

Esta abordagem pode levar a uma distribuição de trabalho desbalanceada se as iterações do loop tiverem trabalho desigual ou `max` não for facilmente divisível pelo número de threads para o loop externo.

#### 2. Colapsando loops aninhados usando a cláusula `collapse(inteiro)`:

```c
#pragma omp parallel for collapse (2) // Mescla dois loops aninhados em um para paralelização
for (int i = 1; i <= max; i++)
    for (int j = 1; j <= max; j++)
        printf ("%d: (%d,%d)\n", omp_get_thread_num (), i, j);
```

Isso efetivamente combina o espaço de iteração do número especificado de loops (ex.: 2 para dois loops aninhados) em um único espaço de iteração maior que é então dividido entre threads, frequentemente levando a um melhor balanceamento de carga.

#### 3. Paralelizando cada loop aninhado separadamente:

```c
#pragma omp parallel for
for (int i = 1; i <= max; i++) {
    #pragma omp parallel for // Loop interno também paralelizado
    for (int j = 1; j <= max; j++) {
        printf ("%d: (%d,%d)\n", omp_get_thread_num (), i, j);
    }
}
```

Isso requer que o **paralelismo aninhado** seja habilitado (ex.: `omp_set_nested(1)` ou `OMP_NESTED=true`). Embora crie mais threads e ofereça um particionamento diferente, também incorre em overhead devido à criação e terminação frequente de threads.

## Combinando os Resultados de Iterações Paralelas

Frequentemente, iterações de loop contribuem para uma única solução combinada, requerendo que resultados parciais sejam agregados. Uma abordagem ingênua usando uma única variável compartilhada para soma pode levar a **condições de corrida**, onde o resultado depende do timing de acessos de leitura e escrita por múltiplas threads.

**Soma com Condição de Corrida**
```c
#include <stdio.h>
int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    int sum = 0;
    #pragma omp parallel for
    for (int i = 1; i <= max; i++)
        sum = sum + i; // Condição de corrida: múltiplas threads atualizam 'sum' simultaneamente
    printf ("%d\n", sum);
    return 0;
}
```

**Explicação**: Múltiplas threads tentam atualizar `sum` simultaneamente. Cada thread lê o `sum` atual, adiciona `i`, e escreve de volta. Se essas operações se sobrepõem, algumas atualizações podem ser perdidas, levando a um resultado incorreto.

### Mecanismos para Evitar Condições de Corrida

Para evitar condições de corrida, o OpenMP fornece mecanismos:

#### 1. `#pragma omp critical`

Garante que o bloco estruturado de código que o segue seja executado por apenas uma thread por vez.

```c
#include <stdio.h>
int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    int sum = 0;
    #pragma omp parallel for
    for (int i = 1; i <= max; i++)
        #pragma omp critical // Apenas uma thread pode executar esta linha por vez
        sum = sum + i;
    printf ("%d\n", sum);
    return 0;
}
```

**Explicação**: Isso garante correção ao forçar acesso exclusivo, mas pode tornar o programa lento pois as threads são forçadas a esperar, efetivamente serializando a parte crítica da computação.

#### 2. `#pragma omp atomic`

Força acesso exclusivo a uma localização de armazenamento para uma única operação de atualização simples (como `sum = sum + i`). É tipicamente mais rápido que uma seção crítica para operações simples pois pode usar instruções de leitura-modificação-escrita suportadas por hardware.

```c
#include <stdio.h>
int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    int sum = 0;
    #pragma omp parallel for
    for (int i = 1; i <= max; i++)
        #pragma omp atomic // Garante que 'sum = sum + i' seja uma operação ininterrupta
        sum = sum + i;
    printf ("%d\n", sum);
    return 0;
}
```

**Explicação**: Esta é uma maneira mais otimizada de realizar atualizações simples em variáveis compartilhadas com segurança.

#### 3. Cláusula `reduction(identificador-reducao:lista)`

Esta é a maneira mais eficiente e preferida de combinar resultados de iterações paralelas para operações comuns como soma, produto, min, max, etc. Para cada variável na lista, uma cópia privada é criada para cada thread. As threads realizam computações em suas cópias privadas, e apenas no final da região paralela essas cópias privadas são combinadas na variável compartilhada original usando a operação de redução especificada.

```c
#include <stdio.h>
#include <omp.h>
int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    int sum = 0;
    #pragma omp parallel for reduction (+:sum) // Cada thread obtém uma cópia privada de 'sum'
    for (int n = 1; n <= max; n++)
        sum = sum + n; // Opera na 'sum' privada
    printf ("%d\n", sum); // 'sum' global é atualizada após a região paralela
    return 0;
}
```

**Explicação**: Isso evita condições de corrida fazendo com que cada thread trabalhe em sua própria variável `sum`. O runtime do OpenMP combina eficientemente esses resultados privados na `sum` global no final.

## Distribuindo Iterações Entre Threads

O OpenMP permite que programadores especifiquem como as iterações de um loop paralelo são distribuídas entre threads usando a **cláusula `schedule`**. Isso é crucial para balanceamento de carga, especialmente quando iterações têm custos computacionais variados.

### Estratégias Comuns de Agendamento

- **`schedule(static)`**: Divide iterações em pedaços de tamanho aproximadamente igual, e cada thread recebe no máximo um pedaço. Isso é adequado quando o trabalho de iteração é uniforme.

- **`schedule(static, tamanho_pedaco)`**: Divide iterações em pedaços de `tamanho_pedaco` e os atribui a threads de forma round-robin. Isso pode ajudar a balancear a carga melhor que o `static` simples se o trabalho de iteração varia mas ainda é um tanto previsível.

- **`schedule(dynamic, tamanho_pedaco)`**: Iterações são divididas em pedaços, e threads dinamicamente pegam pedaços de um pool comum conforme terminam seu trabalho atual. Isso é ideal para loops onde o trabalho de iteração varia significativamente e imprevisivelmente, pois se adapta a variações em tempo de execução.

- **`schedule(auto)`**: O compilador e sistema de runtime escolhem a estratégia de agendamento.

- **`schedule(runtime)`**: A estratégia de agendamento é determinada em tempo de execução pela variável shell `OMP_SCHEDULE`.

**Soma com Agendamento em Runtime**
```c
#include <stdio.h>
#include <unistd.h> // Para sleep
#include <omp.h>

int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    long int sum = 0;
    #pragma omp parallel for reduction (+:sum) schedule(runtime) // Agendamento determinado por OMP_SCHEDULE
    for (int i = 1; i <= max; i++) {
        printf ("%2d @ %d\n", i, omp_get_thread_num ());
        sleep (i < 4 ? i + 1 : 1); // Iterações 1,2,3 levam 2,3,4 segundos; outras levam 1s
        sum = sum + i;
    }
    printf ("%ld\n", sum);
    return 0;
}
```

**Explicação**: Quando executado com `OMP_NUM_THREADS=4` e `max=14`, diferentes configurações de `OMP_SCHEDULE` (ex.: `static`, `static,1`, `dynamic,1`) levarão a diferentes distribuições de trabalho e tempos de execução geral devido aos custos de iteração não uniformes. O agendamento `dynamic` geralmente fornece melhor balanceamento de carga para tais casos.

## Os Detalhes de Loops Paralelos e Reduções

Entender o funcionamento interno de loops paralelos e reduções pode ajudar a escrever código mais otimizado. Uma redução, como o exemplo de soma, pode ser implementada manualmente por:

1. Fazer cada thread computar uma soma parcial em um **elemento de array privado** (ex.: `sums[t]` onde `t` é o ID da thread).
2. Após todas as threads completarem suas computações locais, a thread mestre (ou um processo coordenado) soma esses resultados parciais.

**Soma Manual com Redução**
```c
#include <stdio.h>
#include <omp.h>
int main (int argc , char *argv []) {
    int max; sscanf (argv[1], "%d", &max);
    int ts = omp_get_max_threads (); // Obtém máximo de threads
    if (max % ts != 0) return 1;
    int sums[ts]; // Array para armazenar somas locais para cada thread
    #pragma omp parallel // Região paralela
    {
        int t = omp_get_thread_num (); // ID da thread
        int lo = (max / ts) * (t + 0) + 1;
        int hi = (max / ts) * (t + 1) + 0;
        sums[t] = 0; // Inicializa soma local
        for (int i = lo; i <= hi; i++) // Cada thread computa sua sub-soma
            sums[t] = sums[t] + i;
    }
    int sum = 0;
    for (int t = 0; t < ts; t++) sum = sum + sums[t]; // Thread mestre combina resultados
    printf ("%d\n", sum);
    return 0;
}
```

**Explicação**: Esta implementação manual demonstra como a cláusula `reduction` do OpenMP abstrai a necessidade de gerenciar variáveis locais de thread e sua agregação final explicitamente. O loop de agregação final pode ser otimizado para executar em tempo O(log T) para centenas de threads.

## Considerações Adicionais

Embora loops paralelos sejam comuns, alguns problemas são mais adequados para **tarefas paralelas**, onde segmentos independentes de código (tarefas) são definidos e então agendados para execução. O OpenMP fornece `#pragma omp task` para este propósito, que pode ser usado para criar dinamicamente tarefas cujo número e tamanho podem não ser conhecidos antecipadamente. A diretiva `#pragma omp single` pode ser usada dentro de uma região paralela para garantir que um bloco específico de código seja executado por apenas uma thread na equipe, tipicamente usado para criação de tarefas ou inicialização.
