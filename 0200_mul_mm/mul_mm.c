#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_array(int n, float *x) {
  int i;
  for (i = 0; i < n; i++) {
    x[i] = 0.0f;
  }
}

void mul_mm(float *om, const float *m0, const float *m1) {
  int i, j, k;
  for (i = 0; i < 32; i++) {
    for (k = 0; k < 32; k++) {
      for (j = 0; j < 32; j++) {
        om[i*32+j] += m0[i*32+k] * m1[k*32+j];
      }
    }
  }

  return;
}

int main(int argc, char *argv[]) {
  float *om, *m0, *m1;
  clock_t elapsed;

  om = (float *)malloc(32*32*sizeof(float));
  m0 = (float *)malloc(32*32*sizeof(float));
  m1 = (float *)malloc(32*32*sizeof(float));

  init_array(32*32, om);
  init_array(32*32, m0);
  init_array(32*32, m1);

  elapsed = clock();
  mul_mm(om, m0, m1);
  elapsed = clock() - elapsed;
  printf("It took %.6f seconds.\n", (double)elapsed/CLOCKS_PER_SEC);

  free(om);
  free(m0);
  free(m1);

  return 0;
}
