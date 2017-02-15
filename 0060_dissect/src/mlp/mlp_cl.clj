(ns mlp.mlp-cl
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context conf]
  {:w    (vec (map (fn [[h w]]
                     (let [ar-len (* h w)
                           v (map #(/ % ar-len) (range ar-len))]
                       (cl/create-buffer context :f v)))
                   [[3 4] [4 5]]))
   :b    (vec (map (fn [l] (cl/create-buffer context :f (repeat l 0)))
                   [4 5]))
   :z    (vec (map (partial cl/create-buffer context :f)
                   [4 5]))
   :a    (vec (map (partial cl/create-buffer context :f)
                   [4 5]))
   :v    (vec (map (partial cl/create-buffer context :f)
                   [4 5]))
   :wacc (vec (map (fn [[h w]] (cl/create-buffer context :f (* h w)))
                   [[3 4] [4 5]]))
   :bacc (vec (map (partial cl/create-buffer context :f)
                   [4 5]))})

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
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) conf))
    (ref-set mlp-config (vec conf))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn formatv [v]
  (apply str
   (interpose " "
    (map (partial format "%6.2f")
   ; comma separated, for analyzing on Google Sheet
   ;(interpose ","
   ; (map (partial format "%.2f")
         v))))

(defn print-matrix [cl-mem cr cc] ; column count
  (let [strs (map formatv
                  (partition cc
                             (cl/read-float (@cl-env :queue)
                                            cl-mem
                                            (* cr cc))))]
    (doseq [s strs] (println s))))

(defn dump [k i]
  (printf "%s[%d]:\n" (name k) i)
  (let [[cr cc] (case k
                  (:w :wacc) (nthnext @mlp-config i)
                  (:b :bacc :z :a :v) [1 (@mlp-config (+ i 1))]
                  )]
    (print-matrix (get-in @cl-mem [k i])
                  cr cc)))

(defn fw [in]
  (let [{q :queue} @cl-env
        {dense-fw "dense_fw" add "add" sigmoid-fw "sigmoid_fw"} @cl-ker
        {w :w b :b z :z a :a} @cl-mem]
    (cl/callk q dense-fw   nil [4] :m (z 0) :m in :m (w 0) :i 4 :i 3)
    (cl/callk q add        nil [4] :m (z 0) :m (b 0))
    (cl/callk q sigmoid-fw nil [4] :m (a 0) :m (z 0))
    (cl/callk q dense-fw   nil [5] :m (z 1) :m (a 0) :m (w 1) :i 5 :i 4)
    (cl/callk q add        nil [5] :m (z 1) :m (b 1))
    (cl/callk q sigmoid-fw nil [5] :m (a 1) :m (z 1))
    ))

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        {a :a} @cl-mem
        out (cl/read-float q (a 1) 5)
        lbl (cl/read-float q label 5)] 
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out lbl))))

(defn fw-err-subbatch [inputs labels]
  (apply + (map fw-err inputs labels)))

(defn bw
 ([in label] (bw in label false))
 ([in label is-1st?]
  (let [{q :queue} @cl-env
        {add              "add"
         cross-entropy-bw "cross_entropy_bw"
         dense-bw-v       "dense_bw_v"
         sigmoid-bw       "sigmoid_bw"
         dense-bw-m       "dense_bw_m"
         dense-bw-m-ov    "dense_bw_m_ov"} @cl-ker
        {a :a v :v w :w b :b wacc :wacc bacc :bacc} @cl-mem]
    (cl/callk q cross-entropy-bw nil [5] :m (v 1) :m (a 1) :m label :f 0.1)
    (if is-1st?
      (do (cl/callk q dense-bw-m-ov nil [4 5]
           :m (wacc 1) :m (a 0) :m (v 1) :i 5)
          (CL/clEnqueueCopyBuffer q (v 1) (bacc 1)
           0 0 (* 5 Sizeof/cl_float) 0 nil nil))
      (do (cl/callk q dense-bw-m    nil [4 5]
           :m (wacc 1) :m (a 0) :m (v 1) :i 5)
          (cl/callk q add           nil [5] :m (bacc 1) :m (v 1))
          ))
    (cl/callk q dense-bw-v nil [4] :m (v 0) :m (v 1) :m (w 1) :i 5)
    (cl/callk q sigmoid-bw nil [4] :m (v 0) :m (a 0) :m (v 0))
    (if is-1st?
      (do (cl/callk q dense-bw-m-ov nil [3 4] :m (wacc 0) :m in :m (v 0) :i 4)
          (CL/clEnqueueCopyBuffer q (v 0) (bacc 0)
           0 0 (* 4 Sizeof/cl_float) 0 nil nil))
      (do (cl/callk q dense-bw-m    nil [3 4]
           :m (wacc 0) :m in :m (v 0) :i 4)
          (cl/callk q add           nil [4] :m (bacc 0) :m (v 0))
          )))))

(defn run-subbatch [inputs labels]
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do ;(println "input:") (print-matrix (first i) 1 (first @mlp-config))
          ;(println "label:") (print-matrix (first l) 1 (last @mlp-config))
          (fw (first i))
          (bw (first i) (first l) first?)
          (recur (next i) (next l) false)
          )))
  (let [{q :queue} @cl-env
        {sub "sub"} @cl-ker
        {w :w b :b wacc :wacc bacc :bacc} @cl-mem]
    (cl/callk q sub nil [(* 4 5)] :m (w 1) :m (wacc 1))
    (cl/callk q sub nil [     5 ] :m (b 1) :m (bacc 1))
    (cl/callk q sub nil [(* 3 4)] :m (w 0) :m (wacc 0))
    (cl/callk q sub nil [     4 ] :m (b 0) :m (bacc 0))
    ))
