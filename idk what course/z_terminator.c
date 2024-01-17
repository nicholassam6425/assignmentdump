#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

void main(){
    printf("In Child process!\n");
    system("ps -l");

    system("kill -9 $(ps -l| grep -w Z|tr -s ' '|cut -d ' ' -f 5)");

    printf("The updated list of processes and their status is:\n");
    system("ps -l");
}
