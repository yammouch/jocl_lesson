(ns mlp.core
  (:gen-class)
  (:require [mlp.cl :as cl]
            [clojure.pprint])
  (:import  [org.jocl CL Pointer Sizeof cl_event]))

(set! *warn-on-reflection* true)

(defn prepare-mem [ctx n ^floats a0 ^floats a1]
  (cl/let-err err
    [in0 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR (* Sizeof/cl_float n)
          (Pointer/to a0) err)
     in1 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR (* Sizeof/cl_float n)
          (Pointer/to a1) err)]
    {:out (cl/create-buffer ctx :f n)
     :in0 in0 :in1 in1}))

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

(defn init [n a0 a1]
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) n a0 a1))
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

(defn format-profile [prf]
  (apply format "%8.2e %8.2e %8.2e"
   (->> prf
        (partition 2 1)
        (map (fn [[[_ t0] [_ t1]]] (- t1 t0)))
        (map double)
        )))

(defn print-profile [ev]
  (-> ev get-profile format-profile println))
 
(defn prepare-arrays [n]
  (let [a0 ^floats (make-array Float/TYPE n)
        a1 ^floats (make-array Float/TYPE n)
        ar (make-array Float/TYPE n)  ; array of result
        ak (make-array Float/TYPE n)] ; data for kernel
    (loop [i 0]
      (if (<= n i)
        [a0 a1 ar ak]
        (do (aset a0 i (float i))
            (aset a1 i (float (unchecked-add i 1)))
            (recur (unchecked-add i 1))
            )))))

(defn vecsum-host [n ^floats ar ^floats a0 ^floats a1]
  (let [start-time (System/nanoTime)]
    (loop [i 0]
      (if (<= n i)
        (- (System/nanoTime) start-time)
        (do (aset ar i (unchecked-add (aget a0 i) (aget a1 i)))
            (recur (unchecked-add i 1))
            )))))

(defn compare [n mem ^floats ar ^floats ak]
  (let [{q :queue} @cl-env
        start-time    (atom nil)
        read-end-time (atom nil)
        comp-end-time (atom nil)]
    (reset! start-time (System/nanoTime))
    (cl/handle-cl-error
     (CL/clEnqueueReadBuffer q mem CL/CL_TRUE
      0 (* n Sizeof/cl_float) (Pointer/to ak) 0 nil nil))
    (reset! read-end-time (System/nanoTime))
    (loop [i 0]
      (if (<= n i)
        :done
        (if (== (aget ar i) (aget ak i))
          (recur (unchecked-add i 1))
          (printf "Comparison failed at index %d\n" i))))
    (reset! comp-end-time (System/nanoTime))
    (flush)
    [(- @read-end-time @start-time)
     (- @comp-end-time @read-end-time)]))

(defn call-kernel [gws lws n ev]
  (cl/let-err err
    [{q :queue ctx :context} @cl-env
     {k "k"} @cl-ker
     {out :out in0 :in0 in1 :in1} @cl-mem]
    (cl/set-args k :m out :m in0 :m in1 :i n)
    (CL/clEnqueueNDRangeKernel q k (count gws)
     nil
     (long-array gws)
     (if lws (long-array lws) nil)
     0 nil ev)
    (CL/clWaitForEvents 1 (into-array cl_event [ev]))
    (->> ev
         get-profile
         (partition 2 1)
         (map (fn [[[_ t0] [_ t1]]] (- t1 t0)))
         )))

(defn run1 [gws lws ev a0 a1 ar ak]
  (let [n (apply * gws)
        gpu  (call-kernel gws lws n ev)
        host (vecsum-host n ar a0 a1)]
    (println
     (apply str
            (map #(format %1 %2)
                 (concat (repeat 6 "%5d") (repeat 4 "%10d"))
                 (concat gws lws gpu [host])
                 )))))

(defn -main [& _]
  (cl/let-err err
    [conf [{:gws [512 512 16] :lws [ 64 1 1]}
           {:gws [512 512 32] :lws [ 64 1 1]}
           {:gws [512 512 64] :lws [  1 1 1]}
           {:gws [512 512 64] :lws [  2 1 1]}
           {:gws [512 512 64] :lws [  4 1 1]}
           {:gws [512 512 64] :lws [  8 1 1]}
           {:gws [512 512 64] :lws [ 16 1 1]}
           {:gws [512 512 64] :lws [ 32 1 1]}
           {:gws [512 512 64] :lws [ 64 1 1]}
           {:gws [512 512 64] :lws [128 1 1]}
           {:gws [512 512 64] :lws [256 1 1]}]
     n (apply max (map #(apply * (:gws %)) conf))
     [a0 a1 ar ak] (time (prepare-arrays n))
     _ (init n a0 a1)
     ev (CL/clCreateUserEvent (:context @cl-env) err)]
    (doseq [{gws :gws lws :lws} conf]
      (run1 gws lws ev a0 a1 ar ak))
    (CL/clReleaseEvent ev))
  (finalize))
