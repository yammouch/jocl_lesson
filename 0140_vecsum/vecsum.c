#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_array(int n, float x0, float inc, float *x) {
  int i;
  float xtmp;
  for (i = 0, xtmp = x0; i < n; i++, xtmp += inc) {
    x[i] = xtmp;
  }
}

void vecsum(int n, const float *a, const float *b, float *c) {
  int i;
  for (i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void vecsum_unrolled(int n, const float *a, const float *b, float *c) {
  int i;
  for (i = 0; i < n; i += 16) {
    c[i   ] = a[i   ] + b[i   ];
    c[i+ 1] = a[i+ 1] + b[i+ 1];
    c[i+ 2] = a[i+ 2] + b[i+ 2];
    c[i+ 3] = a[i+ 3] + b[i+ 3];
    c[i+ 4] = a[i+ 4] + b[i+ 4];
    c[i+ 5] = a[i+ 5] + b[i+ 5];
    c[i+ 6] = a[i+ 6] + b[i+ 6];
    c[i+ 7] = a[i+ 7] + b[i+ 7];
    c[i+ 8] = a[i+ 8] + b[i+ 8];
    c[i+ 9] = a[i+ 9] + b[i+ 9];
    c[i+10] = a[i+10] + b[i+10];
    c[i+11] = a[i+11] + b[i+11];
    c[i+12] = a[i+12] + b[i+12];
    c[i+13] = a[i+13] + b[i+13];
    c[i+14] = a[i+14] + b[i+14];
    c[i+15] = a[i+15] + b[i+15];
  }
}


void compare(int n, const float *c) {
  int i, ok;
  for (i = 0; i < n; i++) {
    float expc = 2.0*i + 1.0;
    if (c[i] < expc - 0.01 || expc + 0.01 < c[i]) {
      printf("error on comparison at %d\n", i);
      break;
    }
  }
}

#define N (1 << 24)

int main(int argc, char *argv[]) {
  int i, n;
  float *a, *b, *c, *expc;
  clock_t elapsed;

  n = atoi(argv[1]);
  n = 1 << n;

  a = malloc(n*sizeof(float));
  b = malloc(n*sizeof(float));
  c = malloc(n*sizeof(float));

  init_array(n, 0.0, 1.0, a);
  init_array(n, 1.0, 1.0, b);

  elapsed = clock();
  vecsum_unrolled(n, a, b, c);
  elapsed = clock() - elapsed;
  printf("It took %.6f seconds.\n", (double)elapsed/CLOCKS_PER_SEC);

  elapsed = clock();
  vecsum(n, a, b, c);
  elapsed = clock() - elapsed;
  printf("It took %.6f seconds.\n", (double)elapsed/CLOCKS_PER_SEC);

  compare(n, c);

  free(a); free(b); free(c);


  return 0;
}
