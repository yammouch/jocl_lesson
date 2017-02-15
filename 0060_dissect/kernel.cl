__kernel void set0(
 __global float *out) {
  uint i = get_global_id(0);
  out[i] = 0.0f;
}

__kernel void add(
 __global       float *out,
 __global const float *in) {
  uint i = get_global_id(0);
  out[i] += in[i];
}

__kernel void sub(
 __global       float *out,
 __global const float *in) {
  uint i = get_global_id(0);
  out[i] -= in[i];
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

__kernel void dense_bw_m_ov(
 __global       float *m,
 __global const float *in,
 __global const float *out,
                int    m_w) {
  uint i = get_global_id(0), j = get_global_id(1);
  m[i*m_w+j] = in[i]*out[j];
}

__kernel void dense_bw_ofs(
 __global       float *ofs,
 __global const float *out) {
  uint i = get_global_id(0);
  ofs[i] += out[i];
}

__kernel void dense_bw_ofs_ov(
 __global       float *ofs,
 __global const float *out) {
  uint i = get_global_id(0);
  ofs[i] = out[i];
}

__kernel void dense_bw_v(
 __global       float *in,
 __global const float *out,
 __global const float *m,
                int    m_w) {
  uint i = get_global_id(0), j;
  float acc = 0.0f;
  for (j = 0; j < m_w; j++) {
    acc += out[j]*m[i*m_w+j];
  }
  in[i] = acc;
}

__kernel void sigmoid_fw(
 __global       float *out,
 __global const float *in) {
  uint i = get_global_id(0);
  out[i] = 1.0f/(1.0f + exp(-in[i]));
}

__kernel void sigmoid_bw(
 __global       float *in,
 __global const float *out,
 __global const float *out_prop) {
  uint i = get_global_id(0);
  float x = out[i];
  in[i] = x*(1.0f - x)*out_prop[i];
}

__kernel void softmax_fw_step1(
 __global       float *out,
 __global const float *in) {
  uint i = get_global_id(0);
  out[i] = exp(in[i]);
}

__kernel void softmax_fw_step2(
 __global float *out,
          int    len) {
  int i;
  float acc = 0.0f;
  for (i = 0; i < len; i++) {
    acc += out[i];
  }
  out[len] = acc;
}

__kernel void softmax_fw_step3(
 __global       float *out,
                int    len) {
  uint i = get_global_id(0);
  out[i] = out[i] / out[len];
}

__kernel void quadratic_bw(
 __global       float *in,
 __global const float *out,
 __global const float *expc,
                float  learning_rate) {
  uint i = get_global_id(0);
  float x;
  x = out[i];
  in[i] = (x - expc[i])*learning_rate*x*(1.0f - x);
}

__kernel void cross_entropy_bw(
 __global       float *in,
 __global const float *out,
 __global const float *expc,
                float  learning_rate) {
  uint i = get_global_id(0);
  in[i] = (out[i] - expc[i])*learning_rate;
}
