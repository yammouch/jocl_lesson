public class JKernel {
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
}
