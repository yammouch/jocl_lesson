__kernel void mul_mm(
 __global       float *om,
 __global const float *m0,
 __global const float *m1) {
  uint tid = get_local_id(0);

  __local float  m0l[32*32];
  __local float *m0lp;
  __local float  m1l[32*32];
  __local float  oml[32*32];
  __local float *omlp;
  float acc;

  m0l[32* 0 + tid] = m0[tid]; m0 += 32;
  m0l[32* 1 + tid] = m0[tid]; m0 += 32;
  m0l[32* 2 + tid] = m0[tid]; m0 += 32;
  m0l[32* 3 + tid] = m0[tid]; m0 += 32;
  m0l[32* 4 + tid] = m0[tid]; m0 += 32;
  m0l[32* 5 + tid] = m0[tid]; m0 += 32;
  m0l[32* 6 + tid] = m0[tid]; m0 += 32;
  m0l[32* 7 + tid] = m0[tid]; m0 += 32;
  m0l[32* 8 + tid] = m0[tid]; m0 += 32;
  m0l[32* 9 + tid] = m0[tid]; m0 += 32;
  m0l[32*10 + tid] = m0[tid]; m0 += 32;
  m0l[32*11 + tid] = m0[tid]; m0 += 32;
  m0l[32*12 + tid] = m0[tid]; m0 += 32;
  m0l[32*13 + tid] = m0[tid]; m0 += 32;
  m0l[32*14 + tid] = m0[tid]; m0 += 32;
  m0l[32*15 + tid] = m0[tid]; m0 += 32;
  m0l[32*16 + tid] = m0[tid]; m0 += 32;
  m0l[32*17 + tid] = m0[tid]; m0 += 32;
  m0l[32*18 + tid] = m0[tid]; m0 += 32;
  m0l[32*19 + tid] = m0[tid]; m0 += 32;
  m0l[32*20 + tid] = m0[tid]; m0 += 32;
  m0l[32*21 + tid] = m0[tid]; m0 += 32;
  m0l[32*22 + tid] = m0[tid]; m0 += 32;
  m0l[32*23 + tid] = m0[tid]; m0 += 32;
  m0l[32*24 + tid] = m0[tid]; m0 += 32;
  m0l[32*25 + tid] = m0[tid]; m0 += 32;
  m0l[32*26 + tid] = m0[tid]; m0 += 32;
  m0l[32*27 + tid] = m0[tid]; m0 += 32;
  m0l[32*28 + tid] = m0[tid]; m0 += 32;
  m0l[32*29 + tid] = m0[tid]; m0 += 32;
  m0l[32*30 + tid] = m0[tid]; m0 += 32;
  m0l[32*31 + tid] = m0[tid]; m0 += 32;

  m1l[32* 0 + tid] = m1[tid]; m1 += 32;
  m1l[32* 1 + tid] = m1[tid]; m1 += 32;
  m1l[32* 2 + tid] = m1[tid]; m1 += 32;
  m1l[32* 3 + tid] = m1[tid]; m1 += 32;
  m1l[32* 4 + tid] = m1[tid]; m1 += 32;
  m1l[32* 5 + tid] = m1[tid]; m1 += 32;
  m1l[32* 6 + tid] = m1[tid]; m1 += 32;
  m1l[32* 7 + tid] = m1[tid]; m1 += 32;
  m1l[32* 8 + tid] = m1[tid]; m1 += 32;
  m1l[32* 9 + tid] = m1[tid]; m1 += 32;
  m1l[32*10 + tid] = m1[tid]; m1 += 32;
  m1l[32*11 + tid] = m1[tid]; m1 += 32;
  m1l[32*12 + tid] = m1[tid]; m1 += 32;
  m1l[32*13 + tid] = m1[tid]; m1 += 32;
  m1l[32*14 + tid] = m1[tid]; m1 += 32;
  m1l[32*15 + tid] = m1[tid]; m1 += 32;
  m1l[32*16 + tid] = m1[tid]; m1 += 32;
  m1l[32*17 + tid] = m1[tid]; m1 += 32;
  m1l[32*18 + tid] = m1[tid]; m1 += 32;
  m1l[32*19 + tid] = m1[tid]; m1 += 32;
  m1l[32*20 + tid] = m1[tid]; m1 += 32;
  m1l[32*21 + tid] = m1[tid]; m1 += 32;
  m1l[32*22 + tid] = m1[tid]; m1 += 32;
  m1l[32*23 + tid] = m1[tid]; m1 += 32;
  m1l[32*24 + tid] = m1[tid]; m1 += 32;
  m1l[32*25 + tid] = m1[tid]; m1 += 32;
  m1l[32*26 + tid] = m1[tid]; m1 += 32;
  m1l[32*27 + tid] = m1[tid]; m1 += 32;
  m1l[32*28 + tid] = m1[tid]; m1 += 32;
  m1l[32*29 + tid] = m1[tid]; m1 += 32;
  m1l[32*30 + tid] = m1[tid]; m1 += 32;
  m1l[32*31 + tid] = m1[tid]; m1 += 32;

  m0lp = m0l;
  omlp = oml;

  acc = mad(m0lp[ 0], m1l[32* 0 + tid], 0.0f);
  acc = mad(m0lp[ 1], m1l[32* 1 + tid], acc );
  acc = mad(m0lp[ 2], m1l[32* 2 + tid], acc );
  acc = mad(m0lp[ 3], m1l[32* 3 + tid], acc );
  acc = mad(m0lp[ 4], m1l[32* 4 + tid], acc );
  acc = mad(m0lp[ 5], m1l[32* 5 + tid], acc );
  acc = mad(m0lp[ 6], m1l[32* 6 + tid], acc );
  acc = mad(m0lp[ 7], m1l[32* 7 + tid], acc );
  acc = mad(m0lp[ 8], m1l[32* 8 + tid], acc );
  acc = mad(m0lp[ 9], m1l[32* 9 + tid], acc );
  acc = mad(m0lp[10], m1l[32*10 + tid], acc );
  acc = mad(m0lp[11], m1l[32*11 + tid], acc );
  acc = mad(m0lp[12], m1l[32*12 + tid], acc );
  acc = mad(m0lp[13], m1l[32*13 + tid], acc );
  acc = mad(m0lp[14], m1l[32*14 + tid], acc );
  acc = mad(m0lp[15], m1l[32*15 + tid], acc );
  acc = mad(m0lp[16], m1l[32*16 + tid], acc );
  acc = mad(m0lp[17], m1l[32*17 + tid], acc );
  acc = mad(m0lp[18], m1l[32*18 + tid], acc );
  acc = mad(m0lp[19], m1l[32*19 + tid], acc );
  acc = mad(m0lp[20], m1l[32*20 + tid], acc );
  acc = mad(m0lp[21], m1l[32*21 + tid], acc );
  acc = mad(m0lp[22], m1l[32*22 + tid], acc );
  acc = mad(m0lp[23], m1l[32*23 + tid], acc );
  acc = mad(m0lp[24], m1l[32*24 + tid], acc );
  acc = mad(m0lp[25], m1l[32*25 + tid], acc );
  acc = mad(m0lp[26], m1l[32*26 + tid], acc );
  acc = mad(m0lp[27], m1l[32*27 + tid], acc );
  acc = mad(m0lp[28], m1l[32*28 + tid], acc );
  acc = mad(m0lp[29], m1l[32*29 + tid], acc );
  acc = mad(m0lp[30], m1l[32*30 + tid], acc );
  acc = mad(m0lp[31], m1l[32*31 + tid], acc );
  omlp[tid] = acc;
  m0lp += 32;
  omlp += 32;

  for (int i = 1; i < 32; i++) {
    acc = mad(m0lp[ 0], m1l[32* 0 + tid], 0.0f);
    acc = mad(m0lp[ 1], m1l[32* 1 + tid], acc );
    acc = mad(m0lp[ 2], m1l[32* 2 + tid], acc );
    acc = mad(m0lp[ 3], m1l[32* 3 + tid], acc );
    acc = mad(m0lp[ 4], m1l[32* 4 + tid], acc );
    acc = mad(m0lp[ 5], m1l[32* 5 + tid], acc );
    acc = mad(m0lp[ 6], m1l[32* 6 + tid], acc );
    acc = mad(m0lp[ 7], m1l[32* 7 + tid], acc );
    acc = mad(m0lp[ 8], m1l[32* 8 + tid], acc );
    acc = mad(m0lp[ 9], m1l[32* 9 + tid], acc );
    acc = mad(m0lp[10], m1l[32*10 + tid], acc );
    acc = mad(m0lp[11], m1l[32*11 + tid], acc );
    acc = mad(m0lp[12], m1l[32*12 + tid], acc );
    acc = mad(m0lp[13], m1l[32*13 + tid], acc );
    acc = mad(m0lp[14], m1l[32*14 + tid], acc );
    acc = mad(m0lp[15], m1l[32*15 + tid], acc );
    acc = mad(m0lp[16], m1l[32*16 + tid], acc );
    acc = mad(m0lp[17], m1l[32*17 + tid], acc );
    acc = mad(m0lp[18], m1l[32*18 + tid], acc );
    acc = mad(m0lp[19], m1l[32*19 + tid], acc );
    acc = mad(m0lp[20], m1l[32*20 + tid], acc );
    acc = mad(m0lp[21], m1l[32*21 + tid], acc );
    acc = mad(m0lp[22], m1l[32*22 + tid], acc );
    acc = mad(m0lp[23], m1l[32*23 + tid], acc );
    acc = mad(m0lp[24], m1l[32*24 + tid], acc );
    acc = mad(m0lp[25], m1l[32*25 + tid], acc );
    acc = mad(m0lp[26], m1l[32*26 + tid], acc );
    acc = mad(m0lp[27], m1l[32*27 + tid], acc );
    acc = mad(m0lp[28], m1l[32*28 + tid], acc );
    acc = mad(m0lp[29], m1l[32*29 + tid], acc );
    acc = mad(m0lp[30], m1l[32*30 + tid], acc );
    acc = mad(m0lp[31], m1l[32*31 + tid], acc );
    omlp[tid] += acc;
    m0lp += 32;
    omlp += 32;
  }

  om[tid] = oml[32* 0 + tid]; om += 32;
  om[tid] = oml[32* 1 + tid]; om += 32;
  om[tid] = oml[32* 2 + tid]; om += 32;
  om[tid] = oml[32* 3 + tid]; om += 32;
  om[tid] = oml[32* 4 + tid]; om += 32;
  om[tid] = oml[32* 5 + tid]; om += 32;
  om[tid] = oml[32* 6 + tid]; om += 32;
  om[tid] = oml[32* 7 + tid]; om += 32;
  om[tid] = oml[32* 8 + tid]; om += 32;
  om[tid] = oml[32* 9 + tid]; om += 32;
  om[tid] = oml[32*10 + tid]; om += 32;
  om[tid] = oml[32*11 + tid]; om += 32;
  om[tid] = oml[32*12 + tid]; om += 32;
  om[tid] = oml[32*13 + tid]; om += 32;
  om[tid] = oml[32*14 + tid]; om += 32;
  om[tid] = oml[32*15 + tid]; om += 32;
  om[tid] = oml[32*16 + tid]; om += 32;
  om[tid] = oml[32*17 + tid]; om += 32;
  om[tid] = oml[32*18 + tid]; om += 32;
  om[tid] = oml[32*19 + tid]; om += 32;
  om[tid] = oml[32*20 + tid]; om += 32;
  om[tid] = oml[32*21 + tid]; om += 32;
  om[tid] = oml[32*22 + tid]; om += 32;
  om[tid] = oml[32*23 + tid]; om += 32;
  om[tid] = oml[32*24 + tid]; om += 32;
  om[tid] = oml[32*25 + tid]; om += 32;
  om[tid] = oml[32*26 + tid]; om += 32;
  om[tid] = oml[32*27 + tid]; om += 32;
  om[tid] = oml[32*28 + tid]; om += 32;
  om[tid] = oml[32*29 + tid]; om += 32;
  om[tid] = oml[32*30 + tid]; om += 32;
  om[tid] = oml[32*31 + tid];
}
