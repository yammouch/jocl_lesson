(ns jocl-lesson.fft-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context exp2]
  (into {}
        (map (fn [k factor]
               [k
                (cl/create-buffer context
                 (* (bit-shift-left 1 exp2) Sizeof/cl_float factor))])
             [:w :wave :buf0 :buf1 :result]
             [ 1     1     2     2       1])))

(def kernel-source-code (slurp "fft.cl"))

(defn prepare-kernels [context devices]
  (let [program (cl/compile-kernel-source context devices kernel-source-code)]
    {:program program
     :kernels (into {}
                    (map (fn [k name] [k (cl/create-kernel program name)])
                         [:make-w  :step-1st  :step1  :post-process ]
                         ["make_w" "step_1st" "step1" "post_process"]))}))

(defn engine [q
              {make-w :make-w step-1st :step-1st step1 :step1
               post-process :post-process}
              {w :w wave :wave buf0 :buf0 buf1 :buf1 result :result}
              exp2 factor]
  (let [n      (bit-shift-left 1      exp2 )
        n-half (bit-shift-left 1 (dec exp2))
        _ (do (cl/callk q make-w   nil [n-half] :m w :i exp2)
              (cl/callk q step-1st nil [n-half] :m wave :m buf0 :i n-half))
        butterflied
        (loop [i 1, src buf0, dst buf1, w-mask (int 1)]
          (if (<= exp2 i)
            src
            (do (cl/callk q step1 nil [n-half]
                 :m src :m w :m dst :i n-half :i w-mask)
                (recur (inc i) dst src (bit-or (bit-shift-left w-mask 1) 1))
                )))]
    (cl/callk q post-process nil [n]
     :m butterflied :m result :f factor :i exp2)))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ v] @cl-mem] (CL/clReleaseMemObject v))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(def exp2 (ref 12))

(defn init []
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) @exp2))
    (let [{p :program k :kernels}
          (prepare-kernels (@cl-env :context)
                           [(get-in @cl-env [:device :id])])]
      (ref-set cl-prg p)
      (ref-set cl-ker k))))

(defn fft-mag-norm [bytes ofs swing-0db]
  (let [n (bit-shift-left 1 @exp2)]
    (cl/handle-cl-error
     (CL/clEnqueueWriteBuffer (:queue @cl-env) (:wave @cl-mem) CL/CL_TRUE
      0 (* n Sizeof/cl_float)
      (.withByteOffset (Pointer/to bytes) ofs)
      0 nil nil))
    (engine (:queue @cl-env) @cl-ker @cl-mem @exp2
            (/ 2.0 swing-0db n))
    (cl/read-float (:queue @cl-env) (:result @cl-mem) (bit-shift-left 1 @exp2))
    ))
