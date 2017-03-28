public class JKernel {
  public static void mul_mv(
   int cr, int cc, float[] ov, float[] m, float[] v) {
    for (int i = 0, icc = 0; i < cr; i++, icc += cc) {
      float acc = 0.0f;
      for (int j = 0; j < cc; j++) {
        acc += m[icc+j]*v[j];
      }
      ov[i] = acc;
    }
  }
}
