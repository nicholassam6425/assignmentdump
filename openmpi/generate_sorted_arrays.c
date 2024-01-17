#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

int sortfunc(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main()
{
    int ARRAY_SIZE = 1000000000;
    int A[ARRAY_SIZE], B[ARRAY_SIZE];
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        A[i] = rand() % __INT_MAX__;
        B[i] = rand() % __INT_MAX__;
    }
    printf("Arrays generated\n");
    qsort(A, ARRAY_SIZE, sizeof(int), sortfunc);
    qsort(B, ARRAY_SIZE, sizeof(int), sortfunc);
    printf("Arrays sorted\n");
}