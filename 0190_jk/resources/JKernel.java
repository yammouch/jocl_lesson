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

  public static void softmax(int len, float[] ov, float[] v) {
    float max = v[0];
    for (int i = 1; i < len; i++) {
      if (max < v[i]) max = v[i];
    }
    float tmp, sum = 0.0f;
    for (int i = 0; i < len; i++) { // avoids overflow
      tmp = (float)Math.exp(v[i] - max);
      ov[i] = tmp;
      sum += tmp;
    }
    for (int i = 0; i < len; i++) {
      ov[i] /= sum;
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
}
