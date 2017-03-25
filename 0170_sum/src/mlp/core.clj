(ns mlp.core
  (:gen-class)
  (:require [mlp.cl :as cl]
            [clojure.pprint]
            [clojure.java.io])
  (:import  [org.jocl CL NativePointerObject Pointer Sizeof cl_event]))

(set! *warn-on-reflection* true)

; OpenCL wappers

(defn clGetEventProfilingInfo [ev param-name]
  (cl/parse-unsigned-info
   (cl/query #(CL/clGetEventProfilingInfo ev param-name %1 %2 %3))))

(defn clGetProgramInfo [prg param-name]
  (cond 
    (== param-name CL/CL_PROGRAM_BINARIES)
    (let [body (mapv #(byte-array %)
                     (clGetProgramInfo prg CL/CL_PROGRAM_BINARY_SIZES))]
      (cl/ret-err (CL/clGetProgramInfo prg CL/CL_PROGRAM_BINARIES
                   (apply max (map count body))
                   (Pointer/to (into-array NativePointerObject
                                           (map #(Pointer/to %) body)))
                   nil))
      body)
    (== param-name CL/CL_PROGRAM_BINARY_SIZES)
    (cl/parse-size-t-array
     (cl/query #(CL/clGetProgramInfo prg
                 CL/CL_PROGRAM_BINARY_SIZES %1 %2 %3)))))
     ;(cl/clGetProgramInfo @cl-prg CL/CL_PROGRAM_BINARY_SIZES))))

(defn clGetDeviceInfo [dev param-name]
  (cl/parse-unsigned-info ; it may differ from param-name to param-name
   (cl/query #(CL/clGetDeviceInfo dev param-name %1 %2 %3))
   ))

(defn clGetPlatformInfo [platform param-name]
  (cl/parse-str-info
   (cl/query #(CL/clGetPlatformInfo platform param-name %1 %2 %3))))

; subroutines which do not refer global variables

(defn prepare-mem [ctx n m ^ints a0]
  (cl/let-err err
    [in0 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR (* Sizeof/cl_int n)
          (Pointer/to a0) err)
     out (CL/clCreateBuffer ctx CL/CL_MEM_READ_WRITE (* Sizeof/cl_int m)
          nil err)]
    {:out out :in0 in0}))

(defn get-profile [ev]
  (map (fn [name]
         [name
          (clGetEventProfilingInfo ev
           (.get (.getField CL (str "CL_PROFILING_COMMAND_" name)) nil))])
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

(defn make-test-config [dev size]
  (let [max-group-size (clGetDeviceInfo dev CL/CL_DEVICE_MAX_WORK_GROUP_SIZE)]
    (filter #(<= (:lws %) max-group-size)
            [{:gws size :lws 1024}
             {:gws size :lws  512}
             {:gws size :lws  256}
             {:gws size :lws  128}
             {:gws size :lws   64}])))

; global variables

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))

; followings refer global variables

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
    (ref-set cl-env (cl/context CL/CL_DEVICE_TYPE_GPU))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(:device @cl-env)] kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn call-kernel [gws lws ev]
  (cl/let-err err
    [{q :queue ctx :context} @cl-env
     ;{k "reduceInterleaved"} @cl-ker
     ;{k "reduceCompleteUnrollWarps8"} @cl-ker
     {k "reduceSmemUnroll"} @cl-ker
     {out :out in0 :in0} @cl-mem
     ;read-a (int-array (/ gws lws))]
     read-a (int-array (/ gws lws 8))]
    (cl/set-args k :m in0 :m out :i gws)
    (cl/ret-err
      (CL/clSetKernelArg k 3 (* 1024 Sizeof/cl_int) nil))
    (CL/clEnqueueNDRangeKernel q k 1
     nil
     ;(long-array [gws])
     (long-array [(/ gws 8)])
     (if lws (long-array [lws]) nil)
     0 nil ev)
    (CL/clWaitForEvents 1 (into-array cl_event [ev]))
    (cl/ret-err
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
     conf (do (init)
              (make-test-config (:device @cl-env) size))
     n (apply max (map :gws conf))
     m (apply max (map #(/ (:gws %) (:lws %)) conf))
     [a0 ak] (time (prepare-arrays n m))
     ev (CL/clCreateUserEvent (:context @cl-env) err)]
    (dosync (ref-set cl-mem (prepare-mem (@cl-env :context) n m a0)))
    (when (not= "OpenCL 1.1 ATI-Stream-v2.3 (451)"
                (clGetPlatformInfo (:platform @cl-env) CL/CL_PLATFORM_VERSION))
      (println "dumping...")
      (with-open [o (clojure.java.io/output-stream "kernel.bin")]
        (let [ar (first (clGetProgramInfo @cl-prg CL/CL_PROGRAM_BINARIES))]
          (.write o ar 0 (count ar)))))
    (doseq [{gws :gws lws :lws} conf]
      (run1 gws lws ev a0 ak))
    (CL/clReleaseEvent ev))
  (finalize))
