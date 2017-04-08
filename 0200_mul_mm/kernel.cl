__kernel void mul_mm(
 __global       float *om,
 __global const float *m0,
 __global const float *m1) {
  uint i = get_global_id(0);

  om[32* 0 + i] = m0[ 0] * m1[i];
  om[32* 1 + i] = m0[ 1] * m1[i];
  om[32* 2 + i] = m0[ 2] * m1[i];
  om[32* 3 + i] = m0[ 3] * m1[i];
  om[32* 4 + i] = m0[ 4] * m1[i];
  om[32* 5 + i] = m0[ 5] * m1[i];
  om[32* 6 + i] = m0[ 6] * m1[i];
  om[32* 7 + i] = m0[ 7] * m1[i];
  om[32* 8 + i] = m0[ 8] * m1[i];
  om[32* 9 + i] = m0[ 9] * m1[i];
  om[32*10 + i] = m0[10] * m1[i];
  om[32*11 + i] = m0[11] * m1[i];
  om[32*12 + i] = m0[12] * m1[i];
  om[32*13 + i] = m0[13] * m1[i];
  om[32*14 + i] = m0[14] * m1[i];
  om[32*15 + i] = m0[15] * m1[i];
  om[32*16 + i] = m0[16] * m1[i];
  om[32*17 + i] = m0[17] * m1[i];
  om[32*18 + i] = m0[18] * m1[i];
  om[32*19 + i] = m0[19] * m1[i];
  om[32*20 + i] = m0[20] * m1[i];
  om[32*21 + i] = m0[21] * m1[i];
  om[32*22 + i] = m0[22] * m1[i];
  om[32*23 + i] = m0[23] * m1[i];
  om[32*24 + i] = m0[24] * m1[i];
  om[32*25 + i] = m0[25] * m1[i];
  om[32*26 + i] = m0[26] * m1[i];
  om[32*27 + i] = m0[27] * m1[i];
  om[32*28 + i] = m0[28] * m1[i];
  om[32*29 + i] = m0[29] * m1[i];
  om[32*30 + i] = m0[30] * m1[i];
  om[32*31 + i] = m0[31] * m1[i];
}
