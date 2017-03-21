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

__kernel void reduceCompleteUnrollWarps8(
 __global int *g_idata,
 __global int *g_odata,
          int  n) {
  // set thread ID
  uint tid = get_local_id(0);
  uint idx = get_group_id(0)*get_local_size(0)*8 + get_local_id(0);

  // convert global data pointer to the local pointer of this block
  __global int *idata = g_idata + get_group_id(0)*get_local_size(0)*8;

  // unrolling 8
  if (idx + 7*get_local_size(0) < n) {
    int a1 = g_idata[idx                      ];
    int a2 = g_idata[idx +   get_local_size(0)];
    int a3 = g_idata[idx + 2*get_local_size(0)];
    int a4 = g_idata[idx + 3*get_local_size(0)];
    int b1 = g_idata[idx + 4*get_local_size(0)];
    int b2 = g_idata[idx + 5*get_local_size(0)];
    int b3 = g_idata[idx + 6*get_local_size(0)];
    int b4 = g_idata[idx + 7*get_local_size(0)];
    g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  // in-place reduction in global memory
  if (get_local_size(0) >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (get_local_size(0) >=  512 && tid < 256) idata[tid] += idata[tid + 256];
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (get_local_size(0) >=  256 && tid < 128) idata[tid] += idata[tid + 128];
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (get_local_size(0) >=  128 && tid <  64) idata[tid] += idata[tid +  64];
  barrier(CLK_GLOBAL_MEM_FENCE);

  // unrolling warp
  if (tid < 32) {
    __global volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid +  8];
    vsmem[tid] += vsmem[tid +  4];
    vsmem[tid] += vsmem[tid +  2];
    vsmem[tid] += vsmem[tid +  1];
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[get_group_id(0)] = idata[0];
}


