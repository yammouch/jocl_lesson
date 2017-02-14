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
        {dense-fw "dense_fw" sigmoid-fw "sigmoid_fw"} @cl-ker
        {w :w b :b z :z a :a} @cl-mem]
    (dotimes [i (- (count @mlp-config) 1)]
      (cl/callk q dense-fw   nil [(@mlp-config (+ i 1))]
       :m (z i) :m (if (= i 0) in (a (- i 1))) :m (b i) :m (w i)
       :i (@mlp-config (+ i 1)) :i (@mlp-config i))
      ;(dump :z i)
      (cl/callk q sigmoid-fw nil [(@mlp-config (+ i 1))]
       :m (a i) :m (z i))
      ;(dump :a i)
      )))

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
         cross-entropy-bw "cross_entropy_bw"
         dense-bw-v       "dense_bw_v"
         sigmoid-bw       "sigmoid_bw"
         dense-bw-m       "dense_bw_m"
         dense-bw-m-ov    "dense_bw_m_ov"} @cl-ker
        {a :a v :v w :w b :b wacc :wacc bacc :bacc} @cl-mem
        loop-init (- (count @mlp-config) 2)]
    (doseq [i (range loop-init -1 -1)]
      (if (= i loop-init)
        (cl/callk q cross-entropy-bw nil [(@mlp-config (+ i 1))]
         :m (v i) :m (a i) :m label :f 0.1)
        (do
          (cl/callk q dense-bw-v nil [(@mlp-config (+ i 1))]
           :m (v i) :m (v (+ i 1)) :m (w (+ i 1)) :i (@mlp-config (+ i 2)))
          ;(dump :v i)
          (cl/callk q sigmoid-bw nil [(@mlp-config (+ i 1))]
           :m (v i) :m (a i) :m (v i))))
      ;(dump :v i)
      (if is-1st?
        (do (cl/callk q dense-bw-m-ov nil (take 2 (nthnext @mlp-config i))
             :m (wacc i) :m (if (<= i 0) in (a (- i 1))) :m (v i)
             :i (@mlp-config (+ i 1)))
            (CL/clEnqueueCopyBuffer q (v i) (bacc i)
             0 0 (* (@mlp-config (+ i 1)) Sizeof/cl_float) 0 nil nil))
        (do (cl/callk q dense-bw-m    nil (take 2 (nthnext @mlp-config i))
             :m (wacc i) :m (if (<= i 0) in (a (- i 1))) :m (v i)
             :i (@mlp-config (+ i 1)))
            (cl/callk q add           nil [(@mlp-config (+ i 1))]
             :m (bacc i) :m (v i)
             )))
      ;(dump :wacc i)
      ;(dump :bacc i)
      ))))

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
    (dotimes [i (- (count @mlp-config) 1)]
      (cl/callk q sub nil [(* (@mlp-config i) (@mlp-config (+ i 1)))]
       :m (w i) :m (wacc i))
      (cl/callk q sub nil [(@mlp-config (+ i 1))] :m (b i) :m (bacc i))
      ))
  ;(dotimes [i (- (count @mlp-config) 1)]
  ;  (dump :w i)
  ;  (dump :b i))
)
