public class JKernel {
  public static void add(
   int len, float[] ov, float[] v0, float[] v1) {
    for (int i = 0; i < len; i++) ov[i] = v0[i] + v1[i];
  }

  public static void sub(
   int len, float regu, float[] ov, float[] v0, float[] v1) {
    for (int i = 0; i < len; i++) ov[i] = regu*v0[i] - v1[i];
  }

  public static void mul_mv(
   int cr, int cc, float[] ov, float[] m, float[] v) {
    for (int i = 0; i < cr; i++) {
      float acc = 0.0f;
      for (int j = 0; j < cc; j++) {
        acc += m[i*cc+j]*v[j];
      }
      ov[i] = acc;
    }
  }

  public static void mul_vm(
   int cr, int cc, float[] ov, float[] v, float[] m) {
    for (int j = 0; j < cc; j++) {
      ov[j] = 0.0f;
    }
    for (int i = 0; i < cr; i++) {
      for (int j = 0; j < cc; j++) {
        ov[j] += v[i]*m[i*cc+j];
      }
    }
  }

  public static void mul_vv(
   int cr, int cc, float[] om, float[] vr, float[] vc, boolean is_acc) {
    for (int i = 0; i < cr; i++) {
      for (int j = 0; j < cc; j++) {
        if (is_acc) om[i*cc+j] += vr[i]*vc[j];
        else        om[i*cc+j]  = vr[i]*vc[j];
      }
    }
  }

  public static void sigmoid_fw(int len, float[] ov, float[] v) {
    for (int i = 0; i < len; i++) {
      ov[i] = 1.0f / (1.0f + (float)Math.exp(-v[i]));
    }
  }

  public static void sigmoid_bw(
   int len, float[] ov, float[] fw_out, float[] back_grad) {
    for (int i = 0; i < len; i++) {
      ov[i] = (1.0f - fw_out[i])*fw_out[i]*back_grad[i];
    }
  }

  public static void softmax(int[] len, float[] ov, float[] v) {
    int boundary[] = new int[len.length+1];
    boundary[0] = 0;
    for (int i = 0, acc = 0; i < len.length; i++) {
      acc += len[i];
      boundary[i+1] = acc;
    }

    for (int i = 0; i < len.length; i++) {
      float max = v[boundary[i]];
      for (int j = boundary[i] + 1; j < boundary[i+1]; j++) {
        if (max < v[j]) max = v[j];
      }
      float tmp, sum = 0.0f;
      for (int j = boundary[i]; j < boundary[i+1]; j++) { // avoids overflow
        tmp = (float)Math.exp(v[j] - max);
        ov[j] = tmp;
        sum += tmp;
      }
      for (int j = boundary[i]; j < boundary[i+1]; j++) {
        ov[j] /= sum;
      }
    }
  }

  public static void quadratic_bw(
   int len, float[] ov, float[] fw_out, float[] expc, float learning_rate) {
    float x;
    for (int i = 0; i < len; i++) {
      x = fw_out[i];
      ov[i] = (x - expc[i])*learning_rate*x*(1.0f - x);
    }
  }

  public static void cross_entropy_bw(
   int len, float[] ov, float[] fw_out, float[] expc, float learning_rate) {
    for (int i = 0; i < len; i++) {
      ov[i] = (fw_out[i] - expc[i])*learning_rate;
    }
  }

  public static void conv_fw(
   int     rh, // height of result
   int     rw, // width  of result
   int     ih, // height of input
   int     iw, // width  of input
   int     id, // depth  of input
   int     ch, // height of coeff
   int     cw, // width  of coeff
   int     cd, // depth  of coeff
   int     pu, // padding upside
   int     pl, // padding left
   float[] result,
   float[] input,
   float[] coeff) {
    for (int ry = 0; ry < rh; ry++) {
      for (int rx = 0; rx < rw; rx++) {
        for (int cz = 0; cz < cd; cz++) {
          float acc = 0.0f;
          for (int cy = 0; cy < ch; cy++) {
            int iy = ry + cy - pu;
            if (0 <= iy && iy < ih) {
              for (int cx = 0; cx < cw; cx++) {
                int ix = rx + cx - pl;
                if (0 <= ix && ix < iw) {
                  for (int iz = 0; iz < id; iz++) {
                    acc += input[ (iy*iw+ix)*id+iz       ]
                        *  coeff[((cy*cw+cx)*id+iz)*cd+cz];
                  }
                }
              }
            }
          }
          result[(ry*rw+rx)*cd+cz] = acc;
        }
      }
    }
  }

  public static void conv_bw_u(
   int     rh, // height of result
   int     rw, // width  of result
   int     ih, // height of input
   int     iw, // width  of input
   int     id, // depth  of input
   int     ch, // height of coeff
   int     cw, // width  of coeff
   int     cd, // depth  of coeff
   int     pu, // padding upside
   int     pl, // padding left
   boolean overwrite,
   float[] result,
   float[] input,
   float[] coeff) {

    for (int ry = 0; ry < rh; ry++) {
      for (int rx = 0; rx < rw; rx++) {
        for (int cz = 0; cz < cd; cz++) {
          for (int iz = 0; iz < id; iz++) {
            float acc = 0.0f;
            for (int cy = 0; cy < ch; cy++) {
              int iy = ry + cy - pu;
              if (0 <= iy && iy < ih) {
                for (int cx = 0; cx < cw; cx++) {
                  int ix = rx + cx - pl;
                  if (0 <= ix && ix < iw) {
                    acc += input[(iy*iw+ix)*id+iz]
                        *  coeff[(cy*cw+cx)*cd+cz];
                  }
                }
              }
            }
            if (overwrite) result[((ry*rw+rx)*id+iz)*cd+cz]  = acc;
            else           result[((ry*rw+rx)*id+iz)*cd+cz] += acc;
          }
        }
      }
    }
  }

  public static void conv_bw_b(
   int     rh,   // height of result
   int     rw,   // width  of result
   int     ih,   // height of input
   int     iw,   // width  of input
   int     id,   // depth  of input
   int     ch,   // height of coeff
   int     cw,   // width  of coeff
   int     cd,   // depth  of coeff
   int     pu,   // padding upside
   int     pl,   // padding left
   float[] result,
   float[] input,
   float[] coeff) {
    for (int ry = 0; ry < rh; ry++) {
      for (int rx = 0; rx < rw; rx++) {
        for (int cz = 0; cz < cd; cz++) {
          float acc = 0.0f;
          for (int cy = 0; cy < ch; cy++) {
            int iy = ry + cy - pu;
            if (0 <= iy && iy < ih) {
              for (int cx = 0; cx < cw; cx++) {
                int ix = rx + cx - pl;
                if (0 <= ix && ix < iw) {
                  for (int iz = 0; iz < id; iz++) {
                    acc += input[ (      iy *iw+      ix )*id+iz       ]
                        *  coeff[(((ch-1-cy)*cw+(cw-1-cx))*cd+cz)*id+iz];
                  }
                }
              }
            }
          }
          result[(ry*rw+rx)*cd+cz] = acc;
        }
      }
    }
  }
}
