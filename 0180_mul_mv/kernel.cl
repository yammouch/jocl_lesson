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

__kernel void mul_mv_local_mem(
 __global       float *prod,
 __global const float *m,
 __global const float *v,
 __local        float *lprod,
 __local        float *lv) {
  uint li = get_local_id(0);
  uint ls = get_local_size(0);
  uint mpos;
  lv[li] = v[li];
  for (uint i = 0; i < ls; i++) {
    mpos = ls*i + li;
    lprod[mpos + (mpos >> 5)] = m[mpos]*lv[li];
    //lprod[mpos] = m[mpos]*lv[li];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stride = (ls >> 1); 0 < stride; stride >>= 1) {
    for (uint i = 0; i < stride; i++) {
      mpos     = ls*li + i;
      uint mstrided = mpos + stride;
      lprod[mpos + (mpos >> 5)] += lprod[mstrided + (mstrided >> 5)];
      //lprod[mpos] += lprod[mstrided + (mstrided >> 5)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  mpos = ls*li;
  prod[li] = lprod[mpos + (mpos >> 5)];
  //prod[li] = lprod[mpos];
}
