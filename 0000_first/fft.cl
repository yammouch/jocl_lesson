unsigned int bit_reverse(unsigned int i, int exp2) {
  i = (i >> 16) | (i << 16);
  i = ((i & 0xFF00FF00) >> 8) | ((i & 0x00FF00FF) << 8);
  i = ((i & 0xF0F0F0F0) >> 4) | ((i & 0x0F0F0F0F) << 4);
  i = ((i & 0xCCCCCCCC) >> 2) | ((i & 0x33333333) << 2);
  i = ((i & 0xAAAAAAAA) >> 1) | ((i & 0x55555555) << 1);
  i = (i >> (32 - exp2));
  return i;
}

__kernel void make_w(
 __global float *result,
 int             exp2) {
  unsigned int i = get_global_id(0);
  float phase;
  phase = 2 * M_PI_F * i / (1 << exp2);

  i = bit_reverse(i, (exp2 - 1));

  result[2*i  ] = cos(phase);
  result[2*i+1] = sin(phase);
}

__kernel void step_1st(
 __global const float *src,
 __global       float *dst,
 int                   n_half) {
  unsigned int i = get_global_id(0);
  unsigned int i_4;
  float src0, src1;
  src0 = src[i       ];
  src1 = src[i+n_half];

  i_4 = i << 2;
  dst[i_4    ] = src0 + src1;
  dst[i_4 + 1] = 0.0f;
  dst[i_4 + 2] = src0 - src1;
  dst[i_4 + 3] = 0.0f;
}

__kernel void step1(
 __global const float *src,
 __global const float *w,
 __global       float *dst,
 int                   n_half,
 int                   w_mask) {
  unsigned int i = get_global_id(0);
  unsigned int i_2, i_n_half, i_n_half_2, i_4, w_index_2;
  float src0_re, src0_im, src1_re, src1_im, src1_w_re, src1_w_im, w_re, w_im;
  i_2 = i << 1;
  i_n_half = i + n_half;
  i_n_half_2 = i_n_half << 1;
  src0_re = src[i_2           ];
  src0_im = src[i_2        + 1];
  src1_re = src[i_n_half_2    ];
  src1_im = src[i_n_half_2 + 1];

  w_index_2 = (i & w_mask) << 1;
  w_re    = w[w_index_2    ];
  w_im    = w[w_index_2 + 1];

  src1_w_re = src1_re * w_re - src1_im * w_im;
  src1_w_im = src1_re * w_im + src1_im * w_re;

  i_4 = i << 2;
  dst[i_4    ] = src0_re + src1_w_re;
  dst[i_4 + 1] = src0_im + src1_w_im;
  dst[i_4 + 2] = src0_re - src1_w_re;
  dst[i_4 + 3] = src0_im - src1_w_im;
}

__kernel void post_process(
 __global const float *src,
 __global       float *dst,
 float                 coeff,
 int                   exp2) {
  unsigned int i = get_global_id(0);
  unsigned int i_rev = bit_reverse(i, exp2);
  unsigned int i_2 = i << 1;
  float re, im, mag;
  re = src[i_2    ];
  im = src[i_2 + 1];
  mag = sqrt(re*re + im*im)*coeff;

  dst[i_rev] = mag;
}
