#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
int main()
{
        void f(int);
        int i;

        signal(SIGINT, f);
        for (i = 0; i < 5; i++)
        {
                printf("hello\n");
                sleep(1);
        }
}

void f(int signum)
{
        int stillin = 1;
        while (stillin)
        {
                printf("Interrupted! OK to quit (y/n)?");
                char resp;
                scanf("%c", &resp);
                if (resp == 'N' || resp == 'n')
                {
                        stillin = 0;
                }
                else if (resp == 'Y' || resp == 'y')
                {
                        int correct = 0;
                        while (!correct)
                        {
                                char *passwd;
                                printf("Please enter the password: ");
                                scanf("%s", passwd);
                                char corpass[7] = "Laurier";
                                if (strlen(passwd) == 7)
                                {
                                        int j;
                                        int k = 0;
                                        for (j = 0; j < 7; j++)
                                        {
                                                if (corpass[j] == passwd[j])
                                                {
                                                        k++;
                                                }
                                        }
                                        if (k == 7)
                                        {
                                                exit(0);
                                        }
                                }
                        }
                }
        }
}