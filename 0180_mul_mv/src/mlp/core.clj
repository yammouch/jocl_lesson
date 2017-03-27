(ns mlp.core
  (:gen-class)
  (:require [mlp.cl :as cl]
            [clojure.pprint]
            [clojure.java.io])
  (:import  [org.jocl CL NativePointerObject Pointer Sizeof cl_event]))

;(set! *warn-on-reflection* true)

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

(defn clGetDeviceInfo [dev param-name]
  (cl/parse-unsigned-info ; it may differ from param-name to param-name
   (cl/query #(CL/clGetDeviceInfo dev param-name %1 %2 %3))
   ))

(defn clGetPlatformInfo [platform param-name]
  (cl/parse-str-info
   (cl/query #(CL/clGetPlatformInfo platform param-name %1 %2 %3))))

; subroutines which do not refer global variables

(defn prepare-mem [ctx cr cc am av]
  (cl/let-err err
    [m  (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR
         (* Sizeof/cl_float cr cc) (Pointer/to am) err)
     v  (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR
         (* Sizeof/cl_float cc) (Pointer/to av) err)
     ov (CL/clCreateBuffer ctx CL/CL_MEM_READ_WRITE
         (* Sizeof/cl_int cr) nil err)]
    {:ov ov :m m :v v}))

(defn get-profile [ev]
  (map (fn [name]
         [name
          (clGetEventProfilingInfo ev
           (.get (.getField CL (str "CL_PROFILING_COMMAND_" name)) nil))])
       '[QUEUED SUBMIT START END]))

(defn prepare-arrays [cr cc]
  (let [am  ^floats (make-array Float/TYPE (* cr cc))
        av  ^floats (make-array Float/TYPE cc)
        aov ^floats (make-array Float/TYPE cr)]
    (loop [i 0]
      (if (<= cr i)
        :done
        (do (loop [j 0]
              (if (<= cc j)
                :done
                (do (aset am (+ (* i cc) j) (/ (float i) (float (+ j 1))))
                    (recur (+ j 1))
                    )))
            (recur (+ i 1)))))
    (loop [j 0]
      (if (<= cc j)
        :done
        (do (aset av j (float (/ (+ j 1))))
            (recur (+ j 1))
            )))
    [am av aov]))

(defn mul-mv-host [cr cc ^floats aov ^floats am ^floats av]
  (let [elapsed (atom 0)]
    (reset! elapsed (System/nanoTime))
    (loop [i 0]
      (if (<= cr i)
        (do
          (reset! elapsed (- (System/nanoTime) @elapsed))
          @elapsed) 
        (do
          (loop [j 0 acc (float 0)]
            (if (<= cc j)
              (aset aov i acc)
              (recur
               (unchecked-add j 1)
               (unchecked-add
                acc
                (unchecked-multiply
                 (aget am (unchecked-add (unchecked-multiply i cc)
                                         j))
                 (aget av j))))))
          (recur (unchecked-add i 1))
          )))))

(defn print-matrix [cc ar]
  (doseq [r (map (partial map (partial format "%10.5f"))
                 (partition cc ar))]
    (println r)))

(defn compare [len ^floats ak ^floats ah]
  ;(print-matrix len ak)
  ;(print-matrix len ah)
  (loop [i 0]
    (if (<= len i)
      ;(println "comparison succeeded")
      :done
      (if (<= -0.01 (aget ah i) 0.01)
        (if (<= -0.01 (aget ah i) 0.01)
          (recur (+ i 1))
          (printf "comparison failed at %d\n" i))
        (if (<= 0.99 (/ (aget ak i) (aget ak i)) 1.01)
          (recur (+ i 1))
          (printf "comparison failed at %d\n" i)
          )))))

(defn make-test-config [dev cr cc]
  [{:gws cc :lws (min 256 cc)}])

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

(defn run1-k [k cr cc ev aov am av]
  (cl/let-err err
    [{q :queue ctx :context} @cl-env
     {ov :ov m :m v :v} @cl-mem
     read-ov (make-array Float/TYPE cr)]
    (cl/ret-err
     (CL/clEnqueueWriteBuffer q m CL/CL_TRUE
      0 (* cr cc Sizeof/cl_float) (Pointer/to am) 0 nil nil))
    (cl/ret-err
     (CL/clEnqueueWriteBuffer q v CL/CL_TRUE
      0 (* cc Sizeof/cl_float) (Pointer/to av) 0 nil nil))
    (cl/ret-err
     (CL/clEnqueueNDRangeKernel q k 1
      ;nil (long-array [(* cr cc)]) (long-array [(min cc 256)]) 0 nil ev))
      nil (long-array [(* cr cc)]) (long-array [(min cc 64)]) 0 nil ev))
    (cl/ret-err
     (CL/clWaitForEvents 1 (into-array cl_event [ev])))
    (cl/ret-err
     (CL/clEnqueueReadBuffer q ov CL/CL_TRUE
      0 (* cr Sizeof/cl_float) (Pointer/to read-ov) 0 nil nil))
    (compare cr read-ov aov)
    (->> ev
         get-profile
         (partition 2 1)
         (map (fn [[[_ t0] [_ t1]]] (- t1 t0)))
         )))

(defn main-loop [cr cc ev aov am av]
  (cl/set-args (@cl-ker "mul_mv_local_mem") :m (:ov @cl-mem)
   :m (:m @cl-mem) :m (:v @cl-mem))
  (cl/ret-err
   (CL/clSetKernelArg (@cl-ker "mul_mv_local_mem") 3
    (* (+ (* 64 64) (quot (* 64 64) 32)) Sizeof/cl_float) nil))
  (cl/ret-err
   (CL/clSetKernelArg (@cl-ker "mul_mv_local_mem") 4
    (* 64 Sizeof/cl_float) nil))
  (println
   (run1-k (@cl-ker "mul_mv_local_mem") cr cc ev aov am av)))

(defn -main [& _]
  (cl/let-err err
    [cr (bit-shift-left 1 6)
     cc (bit-shift-left 1 6)
     conf (do (init)
              (make-test-config (:device @cl-env) cr cc))
     [am av aov] (time (prepare-arrays cr cc))
     ev (CL/clCreateUserEvent (:context @cl-env) err)]
    (dosync (ref-set cl-mem (prepare-mem (@cl-env :context) cr cc am av)))
    (when (not= "OpenCL 1.1 ATI-Stream-v2.3 (451)"
                (clGetPlatformInfo (:platform @cl-env) CL/CL_PLATFORM_VERSION))
      (with-open [o (clojure.java.io/output-stream "kernel.bin")]
        (let [ar (first (clGetProgramInfo @cl-prg CL/CL_PROGRAM_BINARIES))]
          (.write o ar 0 (count ar)))))
    ;(print-matrix cc am)
    ;(print-matrix cc av)
    (printf "host: %d\n" (mul-mv-host cr cc aov am av))
    (main-loop cr cc ev aov am av)
    (CL/clReleaseEvent ev))
  (finalize))
