# OpenMP

### Manipulação de Arquivos
No código abaixo, procuramos a quantidade de ocorrências de um determinado número dentro de um arquivo de números (um por linha) gerados aleatoriamente. 

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

int main() {
    int aim = 3, total = 0, thid, nthreads;
    long file_size;
    const char * filename = "./numbers.in";
    FILE * fd_initial = fopen(filename, "r");
    
    if (fd_initial == NULL) {
        perror("Error opening file");
        return 1;
    }

    fseek(fd_initial, 0, SEEK_END);
    file_size = ftell(fd_initial);
    fclose(fd_initial);

    #pragma omp parallel private(thid) reduction(+:total)
    {
        thid = omp_get_thread_num();
        #pragma omp single
        nthreads = omp_get_num_threads();

        long start_offset = thid * (file_size/nthreads);
        long end_offset = (thid == nthreads - 1) ? file_size : (thid + 1) * (file_size/nthreads);
        FILE * local_fd = fopen(filename, "r");
        
        if (local_fd == NULL) {
            printf("Error handling file at thread %d\n", thid);
        }

        fseek(local_fd, start_offset, SEEK_SET);

        int read_val;
        long current_pos;

        while ((current_pos = ftell(local_fd)) < end_offset && fscanf(local_fd, "%d", &read_val) == 1)
            if (read_val == aim) total++;

        fclose(local_fd);
    }

    printf("Total of %ds: %d\n", aim, total);
    return 0;
}
```
