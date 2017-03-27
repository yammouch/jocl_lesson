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
