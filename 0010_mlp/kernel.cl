__kernel void set0(
 __global float *out) {
  uint i = get_global_id(0);
  out[i] = 0.0f;
}

__kernel void dense_fw(
 __global       float *out,
 __global const float *in,
 __global const float *m,
                int    m_w,
                int    m_h) {
  uint i, j = get_global_id(0);
  float acc = 0.0f;

  for (i = 0; i < m_h; i++) {
    acc += in[i] * m[i*m_w+j];
  }
  out[j] = acc;
}

__kernel void dense_bw_m(
 __global       float *m,
 __global const float *in,
 __global const float *out,
                int    m_w) {
  uint i = get_global_id(0), j = get_global_id(1);
  m[i*m_w+j] += in[i]*out[j];
}

__kernel void sigmoid_fw(
 __global       float *out,
 __global const float *in) {
  uint i = get_global_id(0);
  out[i] = 1.0f/(1.0f + exp(-in[i]));
}
