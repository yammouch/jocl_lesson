(ns jocl-lesson.mlp-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context conf]
  {:w    (vec (map (fn [[h w]] (cl/create-buffer context :f (* h w)))
                   (partition 2 1 conf)))
   :b    (vec (map (partial cl/create-buffer context :f)
                   (next conf)))
   :z    (vec (map (partial cl/create-buffer context :f)
                   (next conf)))
   :a    (vec (map (partial cl/create-buffer context :f)
                   (next conf)))
   :v    [                     (cl/create-buffer context :f (apply max conf))]
   :wacc (vec (map (fn [[h w]] (cl/create-buffer context :f (* h w)))
                   (partition 2 1 conf)))
   :bacc (vec (map (partial cl/create-buffer context :f)
                   (next conf)))})

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))
(def mlp-config (ref []))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ v] @cl-mem]
    (doseq [m v] (CL/clReleaseMemObject m)))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init [conf]
  (dosync
    ;(ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_CPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) conf))
    (ref-set mlp-config (vec conf))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn fw [in]
  (let [{q :queue} @cl-env
        {dense-fw "dense_fw" sigmoid-fw "sigmoid_fw"} @cl-ker
        {w :w b :b z :z a :a} @cl-mem]
    (dotimes [i (- (count @mlp-config) 1)]
      (cl/callk q dense-fw   nil [(@mlp-config (+ i 1))]
       :m (z i) :m in :m (b i) :m (w i)
       :i (@mlp-config (+ i 1)) :i (@mlp-config i))
      (cl/callk q sigmoid-fw nil [(@mlp-config (+ i 1))]
       :m (a i) :m (z i)
       ))))

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        {a :a} @cl-mem
        out (cl/read-float q (a (- (count @mlp-config) 2)) (last @mlp-config))
        lbl (cl/read-float q label (last @mlp-config))]
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
         sigmoid-bw       "sigmoid_bw"
         dense-bw-m       "dense_bw_m"
         dense-bw-m-ov    "dense_bw_m_ov"} @cl-ker
        {a :a v :v w :w b :b wacc :wacc bacc :bacc} @cl-mem
        loop-init (- (count @mlp-config) 2)]
    (doseq [i (range loop-init -1 -1)]
      (if (= i loop-init)
        (cl/callk q cross-entropy-bw nil [(@mlp-config (+ i 1))]
         :m (v 0) :m (a i) :m label :f 0.1)
        (cl/callk q sigmoid-bw nil [(@mlp-config (+ i 1))]
         :m (v 0) :m (a i)))
      (if is-1st?
        (do (cl/callk q dense-bw-m-ov nil (take 2 (nthnext @mlp-config i))
             :m (wacc i) :m (if (<= i 0) in (a (- i 1))) :m (v 0)
             :i (@mlp-config (+ i 1)))
            (CL/clEnqueueCopyBuffer q (v 0) (bacc i)
             0 0 (* (@mlp-config (+ i 1)) Sizeof/cl_float) 0 nil nil))
        (do (cl/callk q dense-bw-m    nil (take 2 (nthnext @mlp-config i))
             :m (wacc i) :m (if (<= i 0) in (a (- i 1))) :m (v 0)
             :i (@mlp-config (+ i 1)))
            (cl/callk q add           nil [(@mlp-config (+ i 1))]
             :m (bacc i) :m (v 0)
             )))))))

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
    (dotimes [i (- (count @mlp-config) 1)]
      (cl/callk q sub nil [(* (@mlp-config i) (@mlp-config (+ i 1)))]
       :m (w i) :m (wacc i))
      (cl/callk q sub nil [(@mlp-config (+ i 1))] :m (b i) :m (bacc i))
      )))
