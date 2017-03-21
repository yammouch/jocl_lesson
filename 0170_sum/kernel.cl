__kernel void reduceInterleaved(
 __global int *g_idata,
 __global int *g_odata,
          int  n) {
  // set thread ID
  uint tid = get_local_id(0);
  uint idx = get_group_id(0)*get_local_size(0) + get_local_id(0);

  // convert global data pointer to the local pointer of this block
  __global int *idata = g_idata + get_group_id(0)*get_local_size(0);

  // boundary check
  if (idx >= n) return;

  // in-place reduction in global memory
  for (int stride = get_local_size(0)/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid+stride];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[get_group_id(0)] = idata[0];
}
