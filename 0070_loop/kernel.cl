__kernel void add(
 __global       float *sum,
 __global const float *v0,
 __global const float *v1) {
  uint i = get_global_id(0);
  sum[i] = v0[i] + v1[i];
}

__kernel void sub(
 __global       float *diff,
 __global const float *v0,
 __global const float *v1) {
  uint i = get_global_id(0);
  diff[i] = v0[i] - v1[i];
}

__kernel void mul_vm(
 __global       float *prod,
 __global const float *v,
 __global const float *m,
                int    cr,   // row count
                int    cc) { // column count
  uint i, j = get_global_id(0);
  float acc = 0.0f;

  for (i = 0; i < cr; i++) {
    acc += v[i] * m[i*cc+j];
  }
  prod[j] = acc;
}

__kernel void mul_vv_acc(
 __global       float *m,
 __global const float *v1,
 __global const float *v2,
                int    cc) { // column count
  uint i = get_global_id(0), j = get_global_id(1);
  m[i*cc+j] += v1[i]*v2[j];
}

__kernel void mul_vv(
 __global       float *m,
 __global const float *v1,
 __global const float *v2,
                int    cc) { // column count
  uint i = get_global_id(0), j = get_global_id(1);
  m[i*cc+j] = v1[i]*v2[j];
}

__kernel void mul_mv(
 __global       float *prod,
 __global const float *m,
 __global const float *v,
                int    cc) { // column count
  uint i = get_global_id(0), j;
  float acc = 0.0f;
  for (j = 0; j < cc; j++) {
    acc += m[i*cc+j]*v[j];
  }
  prod[i] = acc;
}

__kernel void sigmoid_fw(
 __global       float *result,
 __global const float *v) {
  uint i = get_global_id(0);
  result[i] = 1.0f/(1.0f + exp(-v[i]));
}

__kernel void sigmoid_bw(
 __global       float *result,
 __global const float *fw_out,
 __global const float *grad) {
  uint i = get_global_id(0);
  float x = fw_out[i];
  result[i] = x*(1.0f - x)*grad[i];
}

__kernel void softmax_fw_step1(
 __global       float *result,
 __global const float *v) {
  uint i = get_global_id(0);
  result[i] = exp(v[i]);
}

__kernel void softmax_fw_step2(
 __global float *v,
          int    len) {
  int i;
  float acc = 0.0f;
  for (i = 0; i < len; i++) {
    acc += v[i];
  }
  v[len] = acc;
}

__kernel void softmax_fw_step3(
 __global       float *result,
 __global const float *in,
                int    len) {
  uint i = get_global_id(0);
  result[i] = in[i] / in[len];
}

__kernel void quadratic_bw(
 __global       float *result,
 __global const float *fw_out,
 __global const float *expc,
                float  learning_rate) {
  uint i = get_global_id(0);
  float x;
  x = fw_out[i];
  result[i] = (x - expc[i])*learning_rate*x*(1.0f - x);
}

__kernel void cross_entropy_bw(
 __global       float *result,
 __global const float *fw_out,
 __global const float *expc,
                float  learning_rate) {
  uint i = get_global_id(0);
  result[i] = (fw_out[i] - expc[i])*learning_rate;
}

__kernel void conv_fw(
 __global       float *result,
 __global const float *input,
 __global const float *core,
                int    wr,   // width  of result
                int    wi,   // width  of input
                int    wc,   // width  of core
                int    hc) { // height of core
  uint x = get_global_id(0), y = get_global_id(1);
  int xc, yc; // for core
  float acc = 0.0f;

  for (yc = 0; yc < hc; yc++) {
    for (xc = 0; xc < wc; xc++) {
      acc += core[yc*wc+xc] * input[(yc+y)*wi+(xc+x)];
    }
  }
  result[y*wr+x] = acc;
}

__kernel void conv_bw_u(
 __global       float *result,
 __global const float *fw_input,
 __global const float *grad,
                int    wr,   // width  of result
                int    wi,   // width  of fw_input
                int    wg,   // width  of grad
                int    hg) { // height of grad
  uint x = get_global_id(0), y = get_global_id(1);
  int xg, yg; // for grad
  float acc = 0.0f;

  for (yg = 0; yg < hg; yg++) {
    for (xg = 0; xg < wg; xg++) {
      acc += grad[yg*wg+xg] * fw_input[(yg+y)*wi+(xg+x)];
    }
  }

  result[y*wr+x] = acc;
}
