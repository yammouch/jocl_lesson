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
}
