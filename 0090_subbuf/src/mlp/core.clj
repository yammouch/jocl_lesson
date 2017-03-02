(ns mlp.core
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL Sizeof Pointer cl_buffer_region))

(defn prepare-mem [ctx]
  (cl/let-err err
   [m0 (cl/create-buffer ctx :f 9)
    m1 (CL/clCreateSubBuffer m0 CL/CL_MEM_READ_WRITE
        CL/CL_BUFFER_CREATE_TYPE_REGION
        (cl_buffer_region. (* 0 Sizeof/cl_float) (* 5 Sizeof/cl_float))
        err)
    m2 (CL/clCreateSubBuffer m0 CL/CL_MEM_READ_WRITE
        CL/CL_BUFFER_CREATE_TYPE_REGION
        (cl_buffer_region. (* 5 Sizeof/cl_float) (* 4 Sizeof/cl_float))
        err)]
    {:m0 m0 :m1 m1 :m2 m2}
    ))

(def kernel-source-code "
__kernel void k0(__global float *dst) {
  uint i = get_global_id(0);
  dst[i] = (float)i;
}

__kernel void k1(__global float *dst) {
  uint i = get_global_id(0);
  dst[i] = i + 1.0f;
}
")

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ m] @cl-mem] (CL/clReleaseMemObject m))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init []
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context)))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn formatv [v]
  (apply str
   (interpose " "
    (map (partial format "%6.2f")
         v))))

; comma separated, for analyzing on Google Sheet
;(defn formatv [v]
;  (apply str
;   (map (partial format ",%.2f")
;        v)))

(defn print-matrix [cl-mem cr cc] ; column count
  (let [strs (map formatv
                  (partition cc
                             (cl/read-float (@cl-env :queue)
                                            cl-mem
                                            (* cr cc))))]
    (doseq [s strs] (println s))))

(defn -main [& _]
  (init)
  (let [{q :queue} @cl-env
        {k0 "k0" k1 "k1"} @cl-ker
        {m0 :m0 m1 :m1 m2 :m2} @cl-mem]
    (cl/callk q k0 nil [9] :m m0) (print-matrix m0 1 9)
    (cl/callk q k0 nil [4] :m m2) (print-matrix m0 1 9)
    (cl/callk q k1 nil [5] :m m1) (print-matrix m0 1 9))
  (finalize))
