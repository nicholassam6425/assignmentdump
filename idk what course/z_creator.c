
#include <stdlib.h>
# include <unistd.h>
# include <sys/types.h>

void main(){
    pid_t cpid = fork();
    if (cpid>0) sleep(120);
    else exit(1);
}