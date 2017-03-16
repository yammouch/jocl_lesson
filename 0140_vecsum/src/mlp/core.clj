(ns mlp.core
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL cl_event))

(defn prepare-mem [ctx n]
  {:out (cl/create-buffer ctx :f n)
   :in0 (cl/create-buffer ctx :f (range n))
   :in1 (cl/create-buffer ctx :f (range 1 (+ 1 n)))})

(def kernel-source-code "
__kernel void k(
 __global       float *out,
 __global const float *in0,
 __global const float *in1,
                int    n) {
  uint i = get_global_id(0)
         + get_global_size(0)*( get_global_id(1) 
                              + get_global_size(1)*get_global_id(2)
                              );
  if (i < n) out[i] = in0[i] + in1[i];
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

(defn init [n]
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) n))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn formatv [v]
  (apply str
   (interpose " "
    (map (partial format "%16.2e")
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

(defn get-profile [ev]
  (map (fn [name]
         (let [full-name (str "CL_PROFILING_COMMAND_" name)]
           [name
            (cl/parse-unsigned-info
             (cl/clGetAnInfo #(CL/clGetEventProfilingInfo ev %1 %2 %3 %4)
                             full-name))]))
       '[QUEUED SUBMIT START END]))

(defn -main [& _]
  (cl/let-err err
    [n 8
     _ (init n)
     {q :queue ctx :context} @cl-env
     {k "k"} @cl-ker
     {out :out in0 :in0 in1 :in1} @cl-mem
     ev (CL/clCreateUserEvent ctx err)]
    (cl/set-args k :m out :m in0 :m in1 :i n)
    (CL/clEnqueueNDRangeKernel q k 1 nil (long-array [n]) nil
     0 nil ev)
    (CL/clWaitForEvents 1 (into-array cl_event [ev]))
    (println (get-profile ev))
    (print-matrix out n 1))
  (finalize))
