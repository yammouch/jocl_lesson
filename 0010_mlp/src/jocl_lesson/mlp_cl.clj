(ns jocl-lesson.mlp-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context n-out]
  (into {}
        (map (fn [[k size]]
               [k (cl/create-buffer context :f size)])
             [[:i0  [0.0 ]] [:l0  [0.0 0.0 0.0]]
              [:i1  [0.25]] [:l1  [0.0 0.0 0.0]]
              [:i2  [0.75]] [:l2  [1.0 1.0 1.0]]
              [:i3  [1.0 ]] [:l3  [1.0 1.0 1.0]]
              [:w   [-1.0 0.0 1.0]]
              [:b   [ 0.0 0.0 0.0]]
              [:z    n-out]
              [:a    n-out]
              [:v    n-out]
              [:wacc n-out]
              [:bacc n-out]
              ])))

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))
(def n-out  (ref 1))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ v] @cl-mem] (CL/clReleaseMemObject v))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init [no]
  (dosync
    ;(ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_CPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) no))
    (ref-set n-out no)
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn fw [in]
  (let [{q :queue} @cl-env
        {dense-fw "dense_fw" sigmoid-fw "sigmoid_fw"} @cl-ker
        {w :w b :b z :z a :a} @cl-mem]
    (cl/callk q dense-fw   nil [@n-out] :m z :m in :m b :m w :i @n-out :i 1)
    (cl/callk q sigmoid-fw nil [@n-out] :m a :m z)
    ))

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        {a :a} @cl-mem
        out (cl/read-float q a @n-out)
        lbl (cl/read-float q label @n-out)]
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out lbl))))

(defn fw-err-subbatch [inputs labels]
  (apply + (map fw-err inputs labels)))

(defn bw
 ([in label] (bw in label false))
 ([in label is-1st?]
  (let [{q :queue} @cl-env
        {add              "add"
         sub              "sub"
         cross-entropy-bw "cross_entropy_bw"
         dense-bw-m       "dense_bw_m"
         dense-bw-m-ov    "dense_bw_m_ov"} @cl-ker
        {a :a v :v w :w b :b wacc :wacc bacc :bacc} @cl-mem]
    (cl/callk q cross-entropy-bw nil [@n-out] :m v :m a :m label :f 0.1)
    (if is-1st?
      (do (cl/callk q dense-bw-m-ov nil [1 @n-out] :m wacc :m in :m v :i @n-out)
          (CL/clEnqueueCopyBuffer q v bacc
           0 0 (* @n-out Sizeof/cl_float) 0 nil nil))
      (do (cl/callk q dense-bw-m    nil [1 @n-out] :m wacc :m in :m v :i @n-out)
          (cl/callk q add           nil [@n-out] :m bacc :m v)
          )))))

(defn run-subbatch [inputs labels]
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do (fw (first i))
          (bw (first i) (first l) first?)
          (recur (next i) (next l) false)
          )))
  (let [{q :queue} @cl-env
        {sub "sub"} @cl-ker
        {w :w b :b wacc :wacc bacc :bacc} @cl-mem]
    (cl/callk q sub nil [@n-out] :m w :m wacc)
    (cl/callk q sub nil [@n-out] :m b :m bacc)))
