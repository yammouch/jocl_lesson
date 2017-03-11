__kernel void set_val(
 __global float *out,
          float  val) {
  uint i = get_global_id(0);
  out[i] = val;
}

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

__kernel void conv(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  int ix, iy; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          acc += input[iy*iw+ix] * coeff[cy*cw+cx];
        }
      }
    }
  }
  result[ry*rw+rx] = acc;
}

__kernel void conv_acc(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  int ix, iy; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          acc += input[iy*iw+ix] * coeff[cy*cw+cx];
        }
      }
    }
  }
  result[ry*rw+rx] += acc;
}

__kernel void conv_t(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  int ix, iy; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          acc += input[iy*iw+ix] * coeff[(ch-1-cy)*cw+(cw-1-cx)];
        }
      }
    }
  }
  result[ry*rw+rx] = acc;
}

__kernel void conv_t_acc(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  int ix, iy; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          acc += input[iy*iw+ix] * coeff[(ch-1-cy)*cw+(cw-1-cx)];
        }
      }
    }
  }
  result[ry*rw+rx] += acc;
}

__kernel void conv_new_fw(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    id,   // depth  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    cd,   // depth  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  uint cz = get_global_id(2);
  int ix, iy, iz; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          for (iz = 0; iz < id; iz++) {
            acc += input[ (iy*iw+ix)*id+iz       ]
                *  coeff[((cy*cw+cx)*id+iz)*cd+cz];
          }
        }
      }
    }
  }
  result[(ry*rw+rx)*cd+cz] += acc;
}

__kernel void conv_new_bw_u(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    id,   // depth  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    cd,   // depth  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx_iz = get_global_id(0);
  int  rx    = rx_iz / id;       // an index of result
  uint ry    = get_global_id(1); // an index of result
  int cx, cy; // indices of coeff
  uint cz = get_global_id(2);
  int ix, iy, iz; // indices of input
  iz = rx_iz % id;
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          acc += input[(iy*iw+ix)*id+iz]
              *  coeff[(cy*cw+cx)*cd+cz];
        }
      }
    }
  }
  result[((ry*rw+rx)*id+iz)*cd+cz] += acc;
}

__kernel void conv_new_bw_g(
 __global       float *result,
 __global const float *input,
 __global const float *coeff,
                int    rw,   // width  of result
                int    ih,   // height of input
                int    iw,   // width  of input
                int    id,   // depth  of input
                int    ch,   // height of coeff
                int    cw,   // width  of coeff
                int    cd,   // depth  of coeff
                int    pu,   // padding upside
                int    pl) { // padding left
  uint rx = get_global_id(0), ry = get_global_id(1); // indices of result
  int cx, cy; // indices of coeff
  uint cz = get_global_id(2);
  int ix, iy, iz; // indices of input
  float acc = 0.0f;

  for (cy = 0; cy < ch; cy++) {
    iy = ry + cy - pu;
    if (0 <= iy && iy < ih) {
      for (cx = 0; cx < cw; cx++) {
        ix = rx + cx - pl;
        if (0 <= ix && ix < iw) {
          for (iz = 0; iz < id; iz++) {
            acc += input[ (      iy *iw+      ix )*id+iz       ]
                *  coeff[(((ch-1-cy)*cw+(cw-1-cx))*cd+cz)*id+iz];
          }
        }
      }
    }
  }
  result[(ry*rw+rx)*cd+cz] += acc;
}
