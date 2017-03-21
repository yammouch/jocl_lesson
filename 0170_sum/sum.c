#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_array(int n, int *x) {
  int i;
  for (i = 0; i < n; i++) {
    x[i] = 1;
  }
}

int sum(int n, const int *a) {
  int i, acc;
  acc = 0;
  for (i = 0; i < n; i++) acc += a[i];
  return acc;
}

int main(int argc, char *argv[]) {
  int i, n, s;
  int *a;
  clock_t elapsed;

  n = atoi(argv[1]);
  n = 1 << n;

  a = malloc(n*sizeof(int));

  init_array(n, a);

  elapsed = clock();
  s = sum(n, a);
  elapsed = clock() - elapsed;
  printf("%d\n", s);
  printf("It took %.6f seconds.\n", (double)elapsed/CLOCKS_PER_SEC);

  free(a);


  return 0;
}
