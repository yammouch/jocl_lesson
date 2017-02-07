(ns jocl-lesson.mlp-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context n-in n-out]
  {:w    (vec (map (fn [[h w]] (cl/create-buffer context :f (* h w)))
                   (partition 2 1 [n-in n-out])))
   :b    (vec (map (partial cl/create-buffer context :f)
                   [n-out]))
   :z    (vec (map (partial cl/create-buffer context :f)
                   [n-out]))
   :a    (vec (map (partial cl/create-buffer context :f)
                   [n-out]))
   :v    [                     (cl/create-buffer context :f (max n-out))]
   :wacc (vec (map (fn [[h w]] (cl/create-buffer context :f (* h w)))
                   (partition 2 1 [n-in n-out])))
   :bacc (vec (map (partial cl/create-buffer context :f)
                   [n-out]))})

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))
(def n-in   (ref 1))
(def n-out  (ref 1))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ v] @cl-mem]
    (doseq [m v] (CL/clReleaseMemObject m)))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init [ni no]
  (dosync
    ;(ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_CPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) ni no))
    (ref-set n-in  ni)
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
    (cl/callk q dense-fw   nil [@n-out] :m (z 0) :m in :m (b 0) :m (w 0)
     :i @n-out :i @n-in)
    (cl/callk q sigmoid-fw nil [@n-out] :m (a 0) :m (z 0))
    ))

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        {a :a} @cl-mem
        out (cl/read-float q (a 0) @n-out)
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
    (cl/callk q cross-entropy-bw nil [@n-out] :m (v 0) :m (a 0) :m label :f 0.1)
    (if is-1st?
      (do (cl/callk q dense-bw-m-ov nil [@n-in @n-out]
           :m (wacc 0) :m in :m (v 0) :i @n-out)
          (CL/clEnqueueCopyBuffer q (v 0) (bacc 0)
           0 0 (* @n-out Sizeof/cl_float) 0 nil nil))
      (do (cl/callk q dense-bw-m    nil [@n-in @n-out]
           :m (wacc 0) :m in :m (v 0) :i @n-out)
          (cl/callk q add           nil [@n-out] :m (bacc 0) :m (v 0))
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
    (cl/callk q sub nil [(* @n-in @n-out)] :m (w 0) :m (wacc 0))
    (cl/callk q sub nil [@n-out] :m (b 0) :m (bacc 0))))
