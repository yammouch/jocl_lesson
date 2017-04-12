(ns mlp.mlp-jk
  (:gen-class))

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

(defn prepare-mem1 [& args]
  (into {} (map (fn [[k x]] [k (float-array x)])
                (partition 2 args))))

(defn prepare-mem-conv
  [init-p {[h w d] :size [ih iw id] :isize :as l}]
  (prepare-mem1 :i (* ih iw id) :p init-p :u (* h w id d)
   :b (* (conv-oh l) (conv-ow l) d)))

(defn prepare-mem [conf seed]
  (mapv (fn [s {t :type [cr cc] :size :as l}]
          (case t
            :dense (prepare-mem1 :i cr :b cc :p s :u (* cr cc))
            :offset (prepare-mem1 :i cr :b cr :p (repeat cr 0) :u cr)
            :conv (prepare-mem-conv s l)
            :sigmoid (prepare-mem1 :i cr :b cr)
            :softmax (prepare-mem1 :i cr :b cr)
            :cross-entropy (prepare-mem1 :i cr)))
        (initial-param conf seed)
        conf))

(def jk-mem     (ref nil))
(def mlp-config (ref nil))

(defn init
 ([conf] (init conf 1))
 ([conf seed]
  (dosync
    (ref-set jk-mem (prepare-mem conf seed))
    (ref-set mlp-config (vec conf))
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

(defn print-matrix [jk-mem cr cc] ; column count
  (let [strs (map formatv
                  (partition cc jk-mem))]
    (doseq [s strs] (println s))))

(defn dump [layers i k]
  (printf "layer %d name %s:\n" i (name k))
  (let [{[h w d] :size [ih iw id] :isize :as l} (layers i)
        [cr cc] (case (l :type)
                  :dense (cond (#{:u :p} k) [h w]
                               (= k :i)     [1 h]
                               (= k :b)     [1 w])
                  :conv (cond (#{:u :p} k) [       h          (*  w id d)]
                              (= k :i)     [      ih          (* iw id  )]
                              (= k :b)     [(conv-oh l) (* (conv-ow l) d)])
                  :offset        [1 h]
                  :sigmoid       [1 h]
                  :softmax       [1 h]
                  :cross-entropy [1 h])]
    (print-matrix (get-in layers [i k])
                  cr cc)))

(defn fw1-conv
  [{[h w d] :size [ih iw id] :isize [pu _ pl _] :pad i :i p :p :as l} {o :i}]
  (let [oh (conv-oh l) ow (conv-ow l)]
    (JKernel/conv_fw oh ow ih iw id h w d pu pl o i p)))

(defn fw1 [{t :type [cr cc] :size i :i p :p :as l} {o :i :as ln}]
  (case t
    :dense   (JKernel/mul_vm              cr  cc o i p)
    :offset  (JKernel/add                 cr     o i p)
    :conv    (fw1-conv l ln)
    :sigmoid (JKernel/sigmoid_fw          cr     o i)
    :softmax (JKernel/softmax (int-array [cr])   o i)
    ))

(defn fw [i0]
  (let [layers (->> (assoc-in @jk-mem [0 :i] i0)
                    (mapv into @mlp-config))]
    (doseq [[l0 l1] (partition 2 1 layers)]
      (fw1 l0 l1))
    (when @debug
      (doseq [i (range (count layers))]
        (dump layers i :i)
        ))))

(defn fw-err [input lbl]
  (fw input)
  (let [out ((last @jk-mem) :i)]
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out lbl))))

(defn fw-err-subbatch [inputs labels]
  (apply + (map fw-err inputs labels)))

(defn bw-dense [{bp :b} {i :i b :b p :p u :u [cr cc] :size} is-1st?]
  (if is-1st?
    (JKernel/mul_vv cr cc u i b false)
    (JKernel/mul_vv cr cc u i b true ))
  (when bp
    (JKernel/mul_mv cr cc bp p b)))

(defn bw-offset [{bp :b} {b :b u :u [n] :size} is-1st?]
  (if is-1st?
    (System/arraycopy b 0 u 0 n)
    (JKernel/add n u u b))
  (when bp
    (System/arraycopy b 0 bp 0 n)))

(defn bw-conv
  [{bp :b}
   {[h w d] :size [ih iw id] :isize [pu _ pl _] :pad
    i :i b :b u :u p :p :as l}
   is-1st?]
  (let [ow (conv-ow l) oh (conv-oh l)]
    (JKernel/conv_bw_u h w ih iw id oh ow d pu pl is-1st? u i b)
    (when bp
      (JKernel/conv_bw_b ih iw oh ow d h w id (- h 1 pu) (- w 1 pl) bp b p)
      )))

(defn bw1
 [{               bp :b            :as lp} ; previous layer
  {t  :type       b  :b [cr] :size :as l }
  {tn :type in :i bn :b                  } ; next layer
  lr ; learning-rate
  is-1st?]
  (case tn
    :cross-entropy
    (case t
      :sigmoid (JKernel/cross_entropy_bw cr bp in bn lr)
      :softmax (JKernel/cross_entropy_bw cr bp in bn lr))
    (case t
      :dense   (bw-dense  lp l is-1st?)
      :offset  (bw-offset lp l is-1st?)
      :conv    (bw-conv   lp l is-1st?)
      :sigmoid (JKernel/sigmoid_bw cr bp in b)
      :softmax (JKernel/sigmoid_bw cr bp in b)
      )))

(defn bw
 ([in label learning-rate] (bw in label false))
 ([i0 label learning-rate is-1st?]
  (let [layers (->> (-> @jk-mem
                        (assoc-in [(- (count @mlp-config) 1) :b] label)
                        (assoc-in [0 :i] i0))
                    (mapv into @mlp-config))]
    (doseq [[lp l ln] (->> (cons nil layers)
                           (partition 3 1)
                           (reverse))]
      (bw1 lp l ln learning-rate is-1st?))
    (when @debug
      (doseq [i (range (- (count layers) 1) -1 -1)]
        (if (get-in layers [i :b]) (dump layers i :b))
        (if (get-in layers [i :u]) (dump layers i :u))
        )))))

(defn run-minibatch
 ([inputs labels] (run-minibatch inputs labels 0.1))
 ([inputs labels learning-rate]
  (when @debug
    (doseq [i (range (count @mlp-config))]
      (when (get-in @jk-mem [i :p])
        (dump (mapv into @mlp-config @jk-mem) i :p))))
  (loop [i inputs l labels first? true]
    (if (or (empty? i) (empty? l))
      :done
      (do (fw (first i))
          (bw (first i) (first l) learning-rate first?)
          (recur (next i) (next l) false)
          )))
  (doseq [{t :type u :u p :p [h w d] :size [_ _ id] :isize}
          (mapv into @mlp-config @jk-mem)]
    (case t
      :dense  (JKernel/sub (* h w     ) p p u)
      :conv   (JKernel/sub (* h w d id) p p u)
      :offset (JKernel/sub    h         p p u)
      :do-nothing))
  (when @debug
    (doseq [i (range (count @mlp-config))]
      (when (get-in @jk-mem [i :p])
        (dump (mapv into @mlp-config @jk-mem) i :p)
        )))))
