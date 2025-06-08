#include <stdio.h>
#include <omp.h>

void selection_sort_parallel(int *v, int n){
    int i;

    for(i = 0; i < n - 1; i++){
        int min = i;

        int local_min = i;

        #pragma omp parallel
        {
            int private_min = local_min;
            int tid = omp_get_thread_num();

            #pragma omp for nowait
            for(int j = i + 1; j < n; j++){
                printf("Thread %d comparando v[%d] = %d\n", tid, j, v[j]);

                if(v[j] < v[private_min])
                    private_min = j;
            }

            #pragma omp critical
            {
                if(v[private_min] < v[local_min])
                    local_min = private_min;
            }
        }

        int tmp = v[i];
        v[i] = v[local_min];
        v[local_min] = tmp;
    }
}

int main() {
    int v[] = {64, 25, 12, 22, 11};
    int n = sizeof(v) / sizeof(v[0]);

    printf("Vetor original:\n");
    for(int i = 0; i < n; i++)
        printf("%d ", v[i]);
    printf("\n");

    selection_sort_parallel(v, n);

    printf("Vetor ordenado:\n");
    for(int i = 0; i < n; i++)
        printf("%d ", v[i]);
    printf("\n");
    
    return 0;
}
