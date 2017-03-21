(ns mlp.core
  (:gen-class)
  (:require [mlp.cl :as cl]
            [clojure.pprint])
  (:import  [org.jocl CL Pointer Sizeof cl_event]))

(set! *warn-on-reflection* true)

(defn prepare-mem [ctx n m ^ints a0]
  (cl/let-err err
    [in0 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR (* Sizeof/cl_int n)
          (Pointer/to a0) err)
     out (CL/clCreateBuffer ctx CL/CL_MEM_READ_WRITE (* Sizeof/cl_int m)
          nil err)]
    {:out out :in0 in0}))

(def kernel-source-code (slurp "kernel.cl"))

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

(defn prepare-arrays [n m]
  (let [a0 ^ints (make-array Integer/TYPE n)
        ak (make-array Integer/TYPE m)] ; data for kernel
    (loop [i 0]
      (if (<= n i)
        [a0 ak]
        (do (aset a0 i 1)
            (recur (unchecked-add i 1))
            )))))

(defn sum-host [n ^ints a0]
  (let [elapsed (atom 0)]
    (reset! elapsed (System/nanoTime))
    (loop [i 0 acc 0]
      (if (<= n i)
        (do (reset! elapsed (- (System/nanoTime) @elapsed))
            (println acc)
            @elapsed) 
        (recur (unchecked-add i 1)
               (unchecked-add acc (aget a0 i))
               )))))

(defn call-kernel [gws lws ev]
  (cl/let-err err
    [{q :queue ctx :context} @cl-env
     ;{k "reduceInterleaved"} @cl-ker
     {k "reduceCompleteUnrollWarps8"} @cl-ker
     {out :out in0 :in0} @cl-mem
     ;read-a (int-array (/ gws lws))]
     read-a (int-array (/ gws lws 8))]
    (cl/set-args k :m in0 :m out :i gws)
    (CL/clEnqueueNDRangeKernel q k 1
     nil
     ;(long-array [gws])
     (long-array [(/ gws 8)])
     (if lws (long-array [lws]) nil)
     0 nil ev)
    (CL/clWaitForEvents 1 (into-array cl_event [ev]))
    (cl/handle-cl-error
     (CL/clEnqueueReadBuffer q out CL/CL_TRUE
      ;0 (* (/ gws lws) Sizeof/cl_int) (Pointer/to read-a) 0 nil nil))
      0 (* (/ gws lws 8) Sizeof/cl_int) (Pointer/to read-a) 0 nil nil))
    (println (apply + read-a))
    (->> ev
         get-profile
         (partition 2 1)
         (map (fn [[[_ t0] [_ t1]]] (- t1 t0)))
         )))

(defn run1 [gws lws ev a0 ak]
  (let [gpu  (call-kernel gws lws ev)
        host (sum-host gws a0)]
    (println
     (apply str
            (map #(format %1 %2)
                 (concat (repeat 2 "%5d") (repeat 4 "%10d"))
                 (concat [gws lws] gpu [host])
                 )))))

(defn -main [& _]
  (cl/let-err err
    [size (bit-shift-left 1 24)
     conf [{:gws size :lws 1024}
           {:gws size :lws  512}
           {:gws size :lws  256}
           {:gws size :lws  128}
           {:gws size :lws   64}]
     n (apply max (map :gws conf))
     m (apply max (map #(/ (:gws %) (:lws %)) conf))
     [a0 ak] (time (prepare-arrays n m))
     _ (init n m a0)
     ev (CL/clCreateUserEvent (:context @cl-env) err)]
    (doseq [{gws :gws lws :lws} conf]
      (run1 gws lws ev a0 ak))
    (CL/clReleaseEvent ev))
  (finalize))
