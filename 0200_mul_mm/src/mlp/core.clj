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

(defn prepare-mem [ctx am0 am1 h-div-32 w-div-32]
  (cl/let-err err
    [m0 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR
         (* Sizeof/cl_float (count am0)) (Pointer/to am0) err)
     m1 (CL/clCreateBuffer ctx CL/CL_MEM_COPY_HOST_PTR
         (* Sizeof/cl_float (count am1)) (Pointer/to am1) err)
     om (CL/clCreateBuffer ctx CL/CL_MEM_READ_WRITE
         (* Sizeof/cl_float 32 h-div-32 32 w-div-32) nil err)]
    {:om om :m0 m0 :m1 m1}))

(defn get-profile [ev]
  (map (fn [name]
         [name
          (clGetEventProfilingInfo ev
           (.get (.getField CL (str "CL_PROFILING_COMMAND_" name)) nil))])
       '[QUEUED SUBMIT START END]))

(defn prepare-arrays [h-div-32 c-div-32 w-div-32]
  (let [aom ^floats (make-array Float/TYPE (* 32 h-div-32 32 w-div-32))
        am0 ^floats (make-array Float/TYPE (* 32 h-div-32 32 c-div-32))
        am1 ^floats (make-array Float/TYPE (* 32 c-div-32 32 w-div-32))]
    (loop [i 0]
      (if (<= (* 32 h-div-32 32 c-div-32) i)
        :done
        (do (aset am0 i (float i))
            (recur (+ i 1)))))
    (loop [i 0]
      (if (<= (* 32 c-div-32 32 w-div-32) i)
        :done
        (do (aset am1 i (float i))
            (recur (+ i 1)))))
    [aom am0 am1]))

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
        (if (<= -0.01 (aget ak i) 0.01)
          (recur (+ i 1))
          (printf "comparison failed at %d\n" i))
        (if (<= 0.99 (/ (aget ak i) (aget ah i)) 1.01)
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

(defn mul-mm-host [^floats aom ^floats am0 ^floats am1
                   ;^long h-div-32 ^long c-div-32 ^long w-div-32]
                   h-div-32 c-div-32 w-div-32]
  (let [start (System/nanoTime)]
    (loop [i 0]
      (if (<= (* 32 h-div-32) i)
        (- (System/nanoTime) start)
        (do (loop [j 0]
              (if (<= (* 32 w-div-32) j)
                :done
                (do (loop [k 0 acc (float 0.0)]
                      (if (<= (* 32 c-div-32) k)
                        (aset aom (+ (* 32 w-div-32 i) j) acc)
                        (recur (+ k 1)
                               (+ acc
                                  (* (aget am0 (+ (* 32 c-div-32 i) k))
                                     (aget am1 (+ (* 32 w-div-32 k) j))
                                     )))))
                    (recur (+ j 1)))))
            (recur (+ i 1)))))))

(defn run1-k [k ev aom am0 am1 h-div-32 c-div-32 w-div-32]
  (let [{q :queue ctx :context} @cl-env
        {om :om m0 :m0 m1 :m1} @cl-mem
        read-om (make-array Float/TYPE (* 32 h-div-32 32 w-div-32))]
    (cl/ret-err
     (CL/clEnqueueWriteBuffer q m0 CL/CL_TRUE
      0 (* 32 h-div-32 32 c-div-32 Sizeof/cl_float) (Pointer/to am0) 0 nil nil)
     (CL/clEnqueueWriteBuffer q m1 CL/CL_TRUE
      0 (* 32 c-div-32 32 w-div-32 Sizeof/cl_float) (Pointer/to am1) 0 nil nil)
     (CL/clEnqueueNDRangeKernel q k 1
      nil (long-array [(* 32 w-div-32) h-div-32]) (long-array [32 1]) 0 nil ev)
     (CL/clWaitForEvents 1 (into-array cl_event [ev]))
     (CL/clEnqueueReadBuffer q om CL/CL_TRUE
      0 (* 32 h-div-32 32 w-div-32 Sizeof/cl_float) (Pointer/to read-om)
      0 nil nil))
    (compare (* 32 h-div-32 32 w-div-32) read-om aom)
    (->> ev
         get-profile
         (partition 2 1)
         (map (fn [[[_ t0] [_ t1]]] (- t1 t0)))
         )))

(defn main-loop [ev aom am0 am1 h-div-32 c-div-32 w-div-32]
  (cl/set-args (@cl-ker "mul_mm") :m (:om @cl-mem)
   :m (:m0 @cl-mem) :m (:m1 @cl-mem)
   :i (* 32 c-div-32) :i c-div-32 :i (- (* 32 c-div-32 31) 32)
   :i (* 32 w-div-32))
  (println
   (run1-k (@cl-ker "mul_mm") ev aom am0 am1 h-div-32 c-div-32 w-div-32)))

(defn -main [& [h-div-32 c-div-32 w-div-32]]
  (init)
  (cl/let-err err
    [h-div-32 (if h-div-32 (read-string h-div-32) 1)
     c-div-32 (if c-div-32 (read-string c-div-32) 1)
     w-div-32 (if w-div-32 (read-string w-div-32) 1)
     [aom am0 am1] (prepare-arrays h-div-32 c-div-32 w-div-32)
     ev (CL/clCreateUserEvent (:context @cl-env) err)]
    (dosync (ref-set cl-mem (prepare-mem (@cl-env :context)
                             am0 am1 h-div-32 w-div-32)))
    (when (not= "OpenCL 1.1 ATI-Stream-v2.3 (451)"
                (clGetPlatformInfo (:platform @cl-env) CL/CL_PLATFORM_VERSION))
      (with-open [o (clojure.java.io/output-stream "kernel.bin")]
        (let [ar (first (clGetProgramInfo @cl-prg CL/CL_PROGRAM_BINARIES))]
          (.write o ar 0 (count ar)))))
    ;(print-matrix cc am)
    ;(print-matrix cc av)
    (printf "host: %d\n" (mul-mm-host aom am0 am1 h-div-32 c-div-32 w-div-32))
    (main-loop ev aom am0 am1 h-div-32 c-div-32 w-div-32)
    (CL/clReleaseEvent ev))
  (finalize))
