(ns mlp.mlp-cl
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL Sizeof Pointer cl_buffer_region))

(defn xorshift [x y z w]
  (let [t  (bit-xor x (bit-shift-left x 11))
        wn (bit-and 0xFFFFFFFF
                    (bit-xor w (bit-shift-right w 19)
                             t (bit-shift-right t  8)))]
    (cons w (lazy-seq (xorshift y z w wn)))))

(defn initial-param [conf seed]
  (loop [[{t :type [h w d] :size [_ _ id] :isize :as c} & cs] conf
         rnd (drop 64 (apply xorshift (range seed (+ seed 4))))
         acc []]
    (if c
      (let [l (case t
                :dense (* h w)
                :conv  (* h w d id)
                0)]
        (recur cs (drop l rnd)
               (conj acc (map #(/ (- (float %) 0x80000000) 0x80000000)
                              (take l rnd)
                              ))))
      acc)))

(defn sub-buffers [m slen]
  (let [mlen (quot (cl/parse-unsigned-info
                    (cl/clGetAnInfo #(CL/clGetMemObjectInfo m %1 %2 %3 %4)
                                    'CL_MEM_SIZE))
                   Sizeof/cl_float)
        loopc (quot mlen slen)]
    (loop [i 0 acc [] ofs 0]
      (if (<= loopc i)
        acc
        (cl/let-err err
         [ms (CL/clCreateSubBuffer m CL/CL_MEM_READ_WRITE
              CL/CL_BUFFER_CREATE_TYPE_REGION
              (cl_buffer_region. (* Sizeof/cl_float ofs)
                                 (* Sizeof/cl_float slen))
              err)]
          (recur (+ i 1) (conj acc ms) (+ ofs slen))
          )))))

(defn conv-oh [{[h _ d] :size [ih _ _] :isize [pu pd _ _] :pad}]
  (* (+ ih (- h) 1 pu pd) d))
(defn conv-ow  [{[_ w d] :size [_ iw _] :isize [_ _ pl pr] :pad}]
  (* (+ iw (- w) 1 pl pr) d))

(defn prepare-mem-conv
  [ctx init-p {[h w d] :size [ih iw id] :isize [pu pd pl pr] :pad :as l}]
  (let [i (cl/create-buffer ctx :f (* ih iw id))
        p (cl/create-buffer ctx :f init-p)
        u (cl/create-buffer ctx :f (* h w id d))
        oh (conv-oh l) ow (conv-ow l)
        g (cl/create-buffer ctx :f (* oh ow d))]
    {:i i :is (sub-buffers i (* ih iw))
     :p p :ps (partition id (sub-buffers i (* h w)))
     :u u :us (partition id (sub-buffers u (* h w)))
     :g g :gs (sub-buffers g (* oh ow))}))

(defn prepare-mem-pass1 [ctx conf seed]
  (mapv (fn [s {t :type [cr cc] :size :as l}]
          (case t
            :dense (into {} (mapv (fn [k x] [k (cl/create-buffer ctx :f x)])
                                  [:i :g :p :u       ]
                                  [cr cc s  (* cr cc)]))
            :offset (into {} (mapv (fn [k x] [k (cl/create-buffer ctx :f x)])
                                   [:i :g :p            :u]
                                   [cr cr (repeat cr 0) cr]))
            :conv (prepare-mem-conv ctx s l)
            :sigmoid (into {} (mapv (fn [k x] [k (cl/create-buffer ctx :f x)])
                                    [:i :g]
                                    [cr cr]))
            :softmax (into {} (mapv (fn [k x] [k (cl/create-buffer ctx :f x)])
                                    [:i :g      ]
                                    [cr (+ cr 1)]))
            :cross-entropy {:i (cl/create-buffer ctx :f cr)}))
        (initial-param conf seed)
        conf))

(defn prepare-mem-pass2 [cl-mem conf]
  (let [c (count conf)]
    (loop [i 0 acc cl-mem]
      (if (<= c i)
        acc
        (recur (+ i 1)
               (let [{[ih iw _] :isize :as l} (conf i)]
                 (if (= :conv (l :type))
                   (let [acc (assoc-in acc [i :os]
                              (sub-buffers (get-in conf [(+ i 1) :i])
                                           (* (conv-oh l) (conv-ow l))))]
                     (if (< 0 i)
                       (assoc-in acc [i :gbs]
                        (sub-buffers (get-in conf [(- i 1) :g])
                                     (* ih iw)))
                       acc))
                   acc)))))))


(def kernel-source-code (slurp "kernel.cl"))

(def cl-env     (ref nil))
(def cl-mem     (ref nil))
(def cl-prg     (ref nil))
(def cl-ker     (ref nil))
(def mlp-config (ref nil))

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

(defn init
 ([conf] (init conf 1))
 ([conf seed]
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (-> (prepare-mem-pass1 (@cl-env :context) conf seed)
                        (prepare-mem-pass2 (vec conf))))
    (ref-set mlp-config (vec conf))
    (ref-set cl-prg (cl/compile-kernel-source (@cl-env :context)
                     [(get-in @cl-env [:device :id])]
                     kernel-source-code))
    (ref-set cl-ker (cl/create-kernels-in-program @cl-prg))
    )))

;(defn formatv [v]
;  (apply str
;   (interpose " "
;    (map (partial format "%6.2f")
;         v))))

; comma separated, for analyzing on Google Sheet
(defn formatv [v]
  (apply str
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
  (let [l (@mlp-config i)
        [cr cc] (l :size)
        [cr cc] (case (l :type)
                  :dense         (cond (#{:u :p} k) [cr cc]
                                       (= k :i)     [ 1 cr]
                                       (= k :g)     [ 1 cc])
                  :offset        [ 1 cr]
                  :sigmoid       [ 1 cr]
                  :softmax       [ 1 cr]
                  :cross-entropy [ 1 cr])]
    (print-matrix (get-in @cl-mem [i k])
                  cr cc)))

(defn fw1-conv
  [{[ih iw id] :isize [h w d] :size [pu pd pl pr] :pad is :is ps :ps os :os
   :as l}]
  (let [{q :queue} @cl-env
        {o "conv" a "conv-acc"} @cl-ker ; o: overwrite, a: accumulate
        oh (conv-oh l) ow (conv-ow l)]
    (doseq [[i j] (for [i (range d) j (range id)] [i j])]
      (cl/callk q (if (= j 0) o a) nil [ow oh]
       :m (os i) :m (is j) :m (get-in ps [i j])
       :i w :i ih :i iw :i h :i w :i pu :i pl
       ))))

(defn fw1 [{t :type [cr cc] :size i :i p :p g :g :as l} {o :i}]
  (let [{q :queue} @cl-env
        {vm "mul_vm" add "add" smd "sigmoid_fw"
         smx1 "softmax_fw_step1" smx2 "softmax_fw_step2"
         smx3 "softmax_fw_step3"} @cl-ker]
    (case t
      :dense       (cl/callk q vm   nil [cc] :m o :m i :m p :i cr :i cc)
      :offset      (cl/callk q add  nil [cr] :m o :m i :m p)
      :conv        (fw1-conv l)
      :sigmoid     (cl/callk q smd  nil [cr] :m o :m i)
      :softmax (do (cl/callk q smx1 nil [cr] :m g :m i)
                   (cl/callk q smx2 nil [ 1] :m g :i cr)
                   (cl/callk q smx3 nil [cr] :m o :m g :i cr)
                   ))))

(defn fw [i0]
  (doseq [[l0 l1] (->> (assoc-in @cl-mem [0 :i] i0)
                       (map into @mlp-config)
                       (partition 2 1))]
    (fw1 l0 l1))
  ;(doseq [i (range 1 (count @mlp-config))]
  ;  (dump i :i))
  )

(defn fw-err [input label]
  (fw input)
  (let [{q :queue} @cl-env
        len (get-in (last @mlp-config) [:size 0])
        a ((last @cl-mem) :i)
        out (cl/read-float q a len)
        lbl (cl/read-float q label len)] 
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out lbl))))

(defn fw-err-subbatch [inputs labels]
  (apply + (map fw-err inputs labels)))

(defn bw-dense [{gp :g} {i :i g :g p :p u :u [cr cc] :size} is-1st?]
  (let [{q :queue} @cl-env
        {vv "mul_vv", vva "mul_vv_acc", mv "mul_mv"} @cl-ker]
    (if is-1st?
      (cl/callk q vv  nil [cr cc] :m u  :m i :m g :i cc)
      (cl/callk q vva nil [cr cc] :m u  :m i :m g :i cc))
    (when gp
      (cl/callk q mv  nil [cr]    :m gp :m p :m g :i cc))))

(defn bw-offset [{gp :g} {g :g u :u [n] :size} is-1st?]
  (let [{q :queue} @cl-env]
    (if is-1st?
      (CL/clEnqueueCopyBuffer q g u 0 0 (* n Sizeof/cl_float) 0 nil nil)
      (cl/callk q (@cl-ker "add") nil [n] :m u :m u :m g))
    (when gp
      (CL/clEnqueueCopyBuffer q g gp 0 0 (* n Sizeof/cl_float) 0 nil nil))))

(defn bw-conv
  [{[w h d] :size [ih iw id] :isize [pu _ pl _] :pad
    is :is gbs :gbs gs :gs us :us ps :ps :as l}
   is-1st?]
  (let [{q :queue} @cl-env
        {o "conv" a "conv-acc" ot "conv-t" at "conv-t-acc"} @cl-ker
        oh (conv-oh l) ow (conv-ow l)]
    (doseq [[i j] (for [i (range d) j (range id)] [i j])]
      (cl/callk q (if is-1st? o a) nil [h w]
       :m (get-in :us [i j]) :m (gs i) :m (is j)
       :i w :i ih :i iw :i oh :i ow :i pu :i pl))
    (doseq [[i j] (for [i (range id) j (range d)] [i j])]
      (cl/callk q (if (= j 0) ot at) nil [ih iw]
       :m (gbs i) :m (:gs j) :m (get-in :ps [j i])
       :i iw :i oh :i ow :i h :i w :i (- h 1 pu) :i (- w 1 pl)
       ))))

(defn bw1
 [{               gp :g            :as lp} ; previous layer
  {t  :type       g  :g [cr] :size :as l }
  {tn :type in :i gn :g                  } ; next layer
  lr ; learning-rate
  is-1st?]
  (let [{q :queue} @cl-env
        {ce "cross_entropy_bw", smd "sigmoid_bw"} @cl-ker]
    (case tn
      :cross-entropy
      (case t
        :sigmoid (cl/callk q ce nil [cr] :m gp :m in :m gn :f lr)
        :softmax (cl/callk q ce nil [cr] :m gp :m in :m gn :f lr))
      (case t
        :dense   (bw-dense  lp l is-1st?)
        :offset  (bw-offset lp l is-1st?)
        :conv    (bw-conv      l is-1st?)
        :sigmoid (cl/callk q smd nil [cr] :m gp :m in :m g)
        :softmax (cl/callk q smd nil [cr] :m gp :m in :m g)
        ))))

(defn bw
 ([in label learning-rate] (bw in label false))
 ([i0 label learning-rate is-1st?]
    (doseq [[lp l ln] (->> (-> @cl-mem
                               (assoc-in [(- (count @mlp-config) 1) :g] label)
                               (assoc-in [0 :i] i0))
                           (map into @mlp-config)
                           (cons nil)
                           (partition 3 1)
                           (reverse))]
      (bw1 lp l ln learning-rate is-1st?))
    ;(doseq [i (range (- (count @mlp-config) 1) -1 -1)]
    ;  (if (get-in @cl-mem [i :g]) (dump i :g))
    ;  (if (get-in @cl-mem [i :u]) (dump i :u)))
    ))

(defn run-minibatch
 ([inputs labels] (run-minibatch inputs labels 0.1))
 ([inputs labels learning-rate]
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do (fw (first i))
          (bw (first i) (first l) learning-rate first?)
          (recur (next i) (next l) false)
          )))
  (let [{q :queue} @cl-env
        {sub "sub"} @cl-ker]
    (doseq [{t :type u :u p :p [cr cc] :size} (mapv into @mlp-config @cl-mem)]
      (case t
        :dense  (cl/callk q sub nil [(* cr cc)] :m p :m p :m u)
        :offset (cl/callk q sub nil [   cr    ] :m p :m p :m u)
        :do-nothing)))))
