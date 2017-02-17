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
  [{:type :dense   :size [3 4]}
   {:type :offset  :size [4  ]}
   {:type :sigmoid :size [4  ]}
   {:type :dense   :size [4 5]}
   {:type :offset  :size [5  ]}
   {:type :sigmoid :size [5  ]}
   {:type :cross-entropy}
   ])

;(require 'clojure.pprint)

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [x @cl-mem]
    ;(clojure.pprint/pprint x)
    (doseq [[_ m] x]
      ;(clojure.pprint/pprint m)
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
  (let [[cr cc] ({              [0 :b] [1 4], [0 :p] [3 4], [0 :u] [3 4],
                  [1 :i] [1 4], [1 :b] [1 4], [1 :p] [1 4], [1 :u] [1 4],
                  [2 :i] [1 4], [2 :b] [1 4],
                  [3 :i] [1 5], [3 :b] [1 5], [3 :p] [4 5], [3 :u] [4 5],
                  [4 :i] [1 5], [4 :b] [1 5], [4 :p] [1 5], [4 :u] [1 5],
                  [5 :i] [1 5]}
                 [i k])]
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
  (let [{q :queue} @cl-env
        {mul-vm "mul_vm" add "add" sigmoid-fw "sigmoid_fw"} @cl-ker
        ;[{      b0 :b p0 :p u0 :u}
        ; {i1 :i b1 :b p1 :p u1 :u}
        ; {i2 :i b2 :b}
        ; {i3 :i b3 :b p3 :p u3 :u}
        ; {i4 :i b4 :b p4 :p u4 :u}
        ; {i5 :i}
        ]; {i6 :i}] @cl-mem]
    (doseq [[l0 l1] (->> (assoc-in @cl-mem [0 :i] i0)
                         (map into mlp-config)
                         (partition 2 1)
                         ;(take 5)
                         )]
      (fw1 l0 l1))
    ;(cl/callk q mul-vm     nil [4] :m i1 :m i0 :m p0 :i 3 :i 4)
    ;(cl/callk q add        nil [4] :m i2 :m i1 :m p1)
    ;(cl/callk q sigmoid-fw nil [4] :m i3 :m i2)
    ;(cl/callk q mul-vm     nil [5] :m i4 :m i3 :m p3 :i 4 :i 5)
    ;(cl/callk q add        nil [5] :m i5 :m i4 :m p4)
    ;(cl/callk q sigmoid-fw nil [5] :m i6 :m i5)
    ;(dump 2 :i) (dump 3 :i) (dump 4 :i) (dump 5 :i) (dump 5 :o)
    ))

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

(defn bw
 ([in label] (bw in label false))
 ([i0 label is-1st?]
  (let [{q :queue} @cl-env
        {add              "add"
         cross-entropy-bw "cross_entropy_bw"
         mul-mv           "mul_mv"
         sigmoid-bw       "sigmoid_bw"
         mul-vv-acc       "mul_vv_acc"
         mul-vv           "mul_vv"} @cl-ker
        [{      b0 :b p0 :p u0 :u}
         {i1 :i b1 :b p1 :p u1 :u}
         {i2 :i b2 :b}
         {i3 :i b3 :b p3 :p u3 :u}
         {i4 :i b4 :b p4 :p u4 :u}
         {i5 :i}
         {i6 :i}] @cl-mem]
    (cl/callk q cross-entropy-bw nil [5] :m b4 :m i6 :m label :f 0.1)
    (if is-1st?
      (CL/clEnqueueCopyBuffer q b4 u4 0 0 (* 5 Sizeof/cl_float) 0 nil nil)
      (cl/callk q add        nil [5]   :m u4 :m u4 :m b4))
    (if is-1st?
      (cl/callk q mul-vv     nil [4 5] :m u3 :m i3 :m b4 :i 5)
      (cl/callk q mul-vv-acc nil [4 5] :m u3 :m i3 :m b4 :i 5))
    (CL/clEnqueueCopyBuffer q b4 b3 0 0 (* 5 Sizeof/cl_float) 0 nil nil)
    (cl/callk q mul-mv     nil [4] :m b2 :m p3 :m b3 :i 5)
    (cl/callk q sigmoid-bw nil [4] :m b1 :m i3 :m b2)
    (if is-1st?
      (CL/clEnqueueCopyBuffer q b1 u1 0 0 (* 4 Sizeof/cl_float) 0 nil nil)
      (cl/callk q add        nil [4]   :m u1 :m u1 :m b1))
    (if is-1st?
      (cl/callk q mul-vv     nil [3 4] :m u0 :m i0 :m b1 :i 4)
      (cl/callk q mul-vv-acc nil [3 4] :m u0 :m i0 :m b1 :i 4))
    ;(dump 4 :b) (dump 4 :u) (dump 3 :u) (dump 2 :b)
    ;(dump 1 :b) (dump 1 :u) (dump 0 :u)
    )))

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
