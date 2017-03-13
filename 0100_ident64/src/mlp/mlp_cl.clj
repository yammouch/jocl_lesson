(ns mlp.mlp-cl
  (:gen-class))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(import '(org.jocl CL Sizeof Pointer cl_buffer_region cl_mem))

(def debug (ref false))

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

(defn conv-oh [{[h _ d] :size [ih _ _] :isize [pu pd _ _] :pad}]
  (+ ih (- h) 1 pu pd))
(defn conv-ow  [{[_ w d] :size [_ iw _] :isize [_ _ pl pr] :pad}]
  (+ iw (- w) 1 pl pr))

(defn prepare-mem1 [ctx & args]
  (into {} (map (fn [[k x]] [k (cl/create-buffer ctx :f x)])
                (partition 2 args))))

(defn prepare-mem-conv
  [ctx init-p {[h w d] :size [ih iw id] :isize :as l}]
  (prepare-mem1 ctx :i (* ih iw id) :p init-p :u (* h w id d)
   :g (* (conv-oh l) (conv-ow l) d)))

(defn prepare-mem [ctx conf seed]
  (mapv (fn [s {t :type [cr cc] :size :as l}]
          (case t
            :dense (prepare-mem1 ctx :i cr :g cc :p s :u (* cr cc))
            :offset (prepare-mem1 ctx :i cr :g cr :p (repeat cr 0) :u cr)
            :conv (prepare-mem-conv ctx s l)
            :sigmoid (prepare-mem1 ctx :i cr :g cr)
            :softmax (prepare-mem1 ctx :i cr :g (+ cr 1))
            :cross-entropy (prepare-mem1 ctx :i cr)))
        (initial-param conf seed)
        conf))

(def kernel-source-code (slurp "kernel.cl"))

(def cl-env     (ref nil))
(def cl-mem     (ref nil))
(def cl-prg     (ref nil))
(def cl-ker     (ref nil))
(def mlp-config (ref nil))

(defn release-mem [x]
  (cond (coll? x) (do (release-mem (first x))
                      (release-mem (next  x)))
        (= (type x) cl_mem) (CL/clReleaseMemObject x)
        :else :do-nothing))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (release-mem @cl-mem)
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init
 ([conf] (init conf 1))
 ([conf seed]
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) conf seed))
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

(defn dump [layers i k]
  (printf "layer %d name %s:\n" i (name k))
  (let [{[h w d] :size [ih iw id] :isize :as l} (layers i)
        [cr cc] (case (l :type)
                  :dense (cond (#{:u :p} k) [h w]
                               (= k :i)     [1 h]
                               (= k :g)     [1 h])
                  :conv (cond (#{:u :p} k) [(*  h d id)        w         ]
                              (= k :i)     [(* ih   id)       iw         ]
                              (= k :g)     [(* (conv-oh l) d) (conv-ow l)])
                  :offset        [1 h]
                  :sigmoid       [1 h]
                  :softmax       [1 h]
                  :cross-entropy [1 h])]
    (print-matrix (get-in layers [i k])
                  cr cc)))

(defn fw1-conv
  [{[h w d] :size [ih iw id] :isize [pu _ pl _] :pad i :i p :p :as l} {o :i}]
  (let [{q :queue} @cl-env
        {k "conv_new_fw"} @cl-ker
        oh (conv-oh l) ow (conv-ow l)]
    (cl/callk q k nil [ow oh d] :m o :m i :m p
     :i ow :i ih :i iw :i id :i h :i w :i d :i pu :i pl
     )))

(defn fw1 [{t :type [cr cc] :size i :i p :p g :g :as l} {o :i :as ln}]
  (let [{q :queue} @cl-env
        {vm "mul_vm" add "add" smd "sigmoid_fw"
         smx1 "softmax_fw_step1" smx2 "softmax_fw_step2"
         smx3 "softmax_fw_step3"} @cl-ker]
    (case t
      :dense       (cl/callk q vm   nil [cc] :m o :m i :m p :i cr :i cc)
      :offset      (cl/callk q add  nil [cr] :m o :m i :m p)
      :conv        (fw1-conv l ln)
      :sigmoid     (cl/callk q smd  nil [cr] :m o :m i)
      :softmax (do (cl/callk q smx1 nil [cr] :m g :m i)
                   (cl/callk q smx2 nil [ 1] :m g :i cr)
                   (cl/callk q smx3 nil [cr] :m o :m g :i cr)
                   ))))

(defn fw [i0]
  (let [layers (->> (assoc-in @cl-mem [0 :i] i0)
                    (mapv into @mlp-config))]
    (doseq [[l0 l1] (partition 2 1 layers)]
      (fw1 l0 l1))
    (when @debug
      (doseq [i (range (count layers))]
        (dump layers i :i)
        ))))

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
  [{gp :g}
   {[h w d] :size [ih iw id] :isize [pu _ pl _] :pad
    i :i g :g u :u p :p :as l}
   is-1st?]
  (let [{q :queue} @cl-env
        {ku "conv_new_bw_u" kg "conv_new_bw_g"} @cl-ker
        ow (conv-ow l) oh (conv-oh l)]
    (cl/callk q ku nil [(* w id) h d] :m u :m i :m g
     :i w :i ih :i iw :i id :i oh :i ow :i d :i pu :i pl
     :i (if is-1st? 1 0))
    (when gp
      (cl/callk q kg nil [iw ih id] :m gp :m g :m p
       :i iw :i oh :i ow :i d :i h :i w :i id :i (- h 1 pu) :i (- w 1 pl)
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
        :conv    (bw-conv   lp l is-1st?)
        :sigmoid (cl/callk q smd nil [cr] :m gp :m in :m g)
        :softmax (cl/callk q smd nil [cr] :m gp :m in :m g)
        ))))

(defn bw
 ([in label learning-rate] (bw in label false))
 ([i0 label learning-rate is-1st?]
  (let [layers (->> (-> @cl-mem
                        (assoc-in [(- (count @mlp-config) 1) :g] label)
                        (assoc-in [0 :i] i0))
                    (mapv into @mlp-config))]
    (doseq [[lp l ln] (->> (cons nil layers)
                           (partition 3 1)
                           (reverse))]
      (bw1 lp l ln learning-rate is-1st?))
    (when @debug
      (doseq [i (range (- (count layers) 1) -1 -1)]
        (if (get-in layers [i :g]) (dump layers i :g))
        (if (get-in layers [i :u]) (dump layers i :u))
        )))))

(defn run-minibatch
 ([inputs labels] (run-minibatch inputs labels 0.1))
 ([inputs labels learning-rate]
  (when @debug
    (doseq [i (range (count @mlp-config))]
      (when (get-in @cl-mem [i :p])
        (dump (mapv into @mlp-config @cl-mem) i :p))))
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do (fw (first i))
          (bw (first i) (first l) learning-rate first?)
          (recur (next i) (next l) false)
          )))
  (let [{q :queue} @cl-env
        {sub "sub"} @cl-ker]
    (doseq [{t :type u :u p :p [h w d] :size [_ _ id] :isize}
            (mapv into @mlp-config @cl-mem)]
      (case t
        :dense  (cl/callk q sub nil [(* h w     )] :m p :m p :m u)
        :conv   (cl/callk q sub nil [(* h w d id)] :m p :m p :m u)
        :offset (cl/callk q sub nil [   h        ] :m p :m p :m u)
        :do-nothing)))
  (when @debug
    (doseq [i (range (count @mlp-config))]
      (when (get-in @cl-mem [i :p])
        (dump (mapv into @mlp-config @cl-mem) i :p)
        )))))
