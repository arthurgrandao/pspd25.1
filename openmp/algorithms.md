# OpenMP

### Soma Paralela

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char * argv[]) {
    long long N = atoi(argv[1]);
    int *v = (int *) malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) v[i] = 1;

    #pragma omp parallel
    {
        for (int shift = 1; shift < N; shift *= 2)
            #pragma omp for
            for (int i = 0; i < N; i += 2 * shift)
                v[i] += v[i + shift];
    }
    printf("Sum is %d\n", v[0]);
    return 0;
}
```
