(ns mlp.mlp-cl
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context]
  [; layer 0, a dense layer
   {:b (cl/create-buffer context :f 4)
    :p (let [ar-len (* 3 4)
             v (map #(/ % ar-len) (range ar-len))]
         (cl/create-buffer context :f v))
    :u (cl/create-buffer context :f (* 3 4))}
   ; layer 1, an offset layer
   {:i (cl/create-buffer context :f 4)
    :b (cl/create-buffer context :f 4)
    :p (cl/create-buffer context :f (repeat 4 0))
    :u (cl/create-buffer context :f 4)}
   ; layer 2, a sigmoid layer
   {:i (cl/create-buffer context :f 4)
    :b (cl/create-buffer context :f 4)}
   ; layer 3, a dense layer
   {:i (cl/create-buffer context :f 4)
    :b (cl/create-buffer context :f 5)
    :p (let [ar-len (* 4 5)
             v (map #(/ % ar-len) (range ar-len))]
         (cl/create-buffer context :f v))
    :u (cl/create-buffer context :f (* 4 5))}
   ; layer 4, an offset layer
   {:i (cl/create-buffer context :f 5)
    :b (cl/create-buffer context :f 5)
    :p (cl/create-buffer context :f (repeat 5 0))
    :u (cl/create-buffer context :f 5)}
   ; layer 5, a sigmoid layer
   {:i (cl/create-buffer context :f 5)}
   ; layer 6, a receiver of the output
   {:i (cl/create-buffer context :f 5)}])

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))
(def mlp-config
  [{:type :dense         :size [3 4]}
   {:type :offset        :size [4  ]}
   {:type :sigmoid       :size [4  ]}
   {:type :dense         :size [4 5]}
   {:type :offset        :size [5  ]}
   {:type :sigmoid       :size [5  ]}
   {:type :cross-entropy :size [5  ]}
   ])

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [x @cl-mem]
    (doseq [[_ m] x]
      (CL/clReleaseMemObject m)))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init [_]
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context)))
    ;(ref-set mlp-config (vec conf))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    ))

(defn formatv [v]
  (apply str
   ;(interpose " "
   ; (map (partial format "%6.2f")
   ; comma separated, for analyzing on Google Sheet
   (map (partial format ",%.2f")
        v)))

(defn print-matrix [cl-mem cr cc] ; column count
  (let [strs (map formatv
                  (partition cc
                             (cl/read-float (@cl-env :queue)
                                            cl-mem
                                            (* cr cc))))]
    (doseq [s strs] (println s))))

(defn dump [i k]
  (printf "layer %d name %s:\n" i (name k))
  (let [l (mlp-config i)
        [cr cc] (l :size)
        [cr cc] (case (l :type)
                  :dense         [cr cc]
                  :offset        [ 1 cr]
                  :sigmoid       [ 1 cr]
                  :cross-entropy [ 1 cr])]
    (print-matrix (get-in @cl-mem [i k])
                  cr cc)))

(defn fw1 [{t :type [cr cc] :size i :i p :p} {o :i}]
  (let [{q :queue} @cl-env]
    (case t
      :dense
      (cl/callk q (@cl-ker "mul_vm")     nil [cc] :m o :m i :m p :i cr :i cc)
      :offset
      (cl/callk q (@cl-ker "add")        nil [cr] :m o :m i :m p)
      :sigmoid
      (cl/callk q (@cl-ker "sigmoid_fw") nil [cr] :m o :m i)
      )))

(defn fw [i0]
  (doseq [[l0 l1] (->> (assoc-in @cl-mem [0 :i] i0)
                       (map into mlp-config)
                       (partition 2 1))]
    (fw1 l0 l1)))

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        a (get-in @cl-mem [6 :i])
        out (cl/read-float q a 5)
        lbl (cl/read-float q label 5)] 
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out lbl))))

(defn fw-err-subbatch [inputs labels]
  (apply + (map fw-err inputs labels)))

(defn bw-dense [{bp :b} {i :i b :b p :p u :u [cr cc] :size} is-1st?]
  (let [{q :queue} @cl-env
        {vv "mul_vv", vva "mul_vv_acc", mv "mul_mv"} @cl-ker]
    (if is-1st?
      (cl/callk q vv  nil [cr cc] :m u  :m i :m b :i cc)
      (cl/callk q vva nil [cr cc] :m u  :m i :m b :i cc))
    (when bp
      (cl/callk q mv  nil [cr]    :m bp :m p :m b :i cc))))

(defn bw-offset [{bp :b} {b :b u :u [n] :size} is-1st?]
  (let [{q :queue} @cl-env]
    (if is-1st?
      (CL/clEnqueueCopyBuffer q b u 0 0 (* n Sizeof/cl_float) 0 nil nil)
      (cl/callk q (@cl-ker "add") nil [n] :m u :m u :m b))
    (when bp
      (CL/clEnqueueCopyBuffer q b bp 0 0 (* n Sizeof/cl_float) 0 nil nil))))

(defn bw1
 [{               bp :b            :as lp} ; previous layer
  {t  :type       b  :b [cr] :size :as l }
  {tn :type in :i bn :b                  } ; next layer
  is-1st?]
  (let [{q :queue} @cl-env
        {ce "cross_entropy_bw", smd "sigmoid_bw"} @cl-ker]
    (case tn
      :cross-entropy
      (case t
        :sigmoid (cl/callk q ce nil [cr] :m bp :m in :m bn :f 0.1))
      (case t
        :dense   (bw-dense  lp l is-1st?)
        :offset  (bw-offset lp l is-1st?)
        :sigmoid (cl/callk q smd nil [4] :m bp :m in :m b)
        ))))

(require 'clojure.pprint)

(defn bw
 ([in label] (bw in label false))
 ([i0 label is-1st?]
    (doseq [[lp l ln] (->> (-> @cl-mem
                               (assoc-in [(- (count mlp-config) 1) :b] label)
                               (assoc-in [0 :i] i0))
                           (map into mlp-config)
                           (cons nil)
                           (partition 3 1)
                           (reverse))]
      (bw1 lp l ln is-1st?))))

(defn run-subbatch [inputs labels]
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do ;(println "input:") (print-matrix (first i) 1 3)
          ;(println "label:") (print-matrix (first l) 1 5)
          (fw (first i))
          (bw (first i) (first l) first?)
          (recur (next i) (next l) false)
          )))
  (let [{q :queue} @cl-env
        {sub "sub"} @cl-ker
        [{p0 :p u0 :u} {p1 :p u1 :u} _
         {p3 :p u3 :u} {p4 :p u4 :u} _] @cl-mem]
    (cl/callk q sub nil [(* 3 4)] :m p0 :m p0 :m u0)
    (cl/callk q sub nil [     4 ] :m p1 :m p1 :m u1)
    (cl/callk q sub nil [(* 4 5)] :m p3 :m p3 :m u3)
    (cl/callk q sub nil [     5 ] :m p4 :m p4 :m u4)
    ))
