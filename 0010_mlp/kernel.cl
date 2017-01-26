__kernel void set0(
 __global float *out) {
  uint i = get_global_id(0);
  out[i] = 0.0f;
}

/*
__kernel void dense_fw(
 __global       float *out,
 __global const float *in,
 __global const float *m,
                int    m_width
) {
  uint i = get_global_id(0);
}
*/
