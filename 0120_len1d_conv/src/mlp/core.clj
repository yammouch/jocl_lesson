; lein run  10 10 4001 0.1 1 30 # converges

(ns mlp.core
  (:gen-class))

(import '(org.jocl CL)
        '(java.util Date))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(require 'mlp.mlp-cl)
(alias 'mlp-cl 'mlp.mlp-cl)

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) (dec i) 1))

(defn a-field [field-size i j]
  (loop [k i acc (vec (repeat field-size 0))]
    (if (<= j k)
      acc
      (recur (inc k) (assoc acc k 1))
      )))

(defn xorshift [x y z w]
  (let [t  (bit-xor x (bit-shift-left x 11))
        wn (bit-and 0xFFFFFFFF
                    (bit-xor w (bit-shift-right w 19)
                             t (bit-shift-right t  8)))]
    (cons w (lazy-seq (xorshift y z w wn)))))

(defn make-input-labels [field-size max-len]
  (let [{ctx :context} @mlp-cl/cl-env
        ij (for [i (range      field-size )
                 j (range (inc field-size)) :when (<= 1 (- j i) max-len)]
             [i j])]
    [(mapv (comp (partial cl/create-buffer ctx :f)
                 (partial apply a-field field-size))
           ij)
     (mapv (comp (partial cl/create-buffer ctx :f)
                 (fn [[i j]] (one-hot max-len (- j i))))
           ij)]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [max-len fs cs cd]
  ; fs: field-size, cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        cosize (* cd (+ fs (if (even? cs) 1 0)))] ; conv out size
    [{:type :conv
      :size  [cs 1 cd]
      :isize [fs 1  1]
      :pad [cs-h 0 cs-h 0]}
     {:type :sigmoid       :size [cosize]}
     {:type :dense         :size [cosize max-len]}
     {:type :offset        :size [max-len]}
     {:type :softmax       :size [max-len]}
     {:type :cross-entropy :size [max-len]}]))

(defn -main [& args]
  (println "start: " (.toString (Date.)))
  (let [[field-size max-len iter learning-rate seed conv-size conv-depth]
        (mapv read-string args)
        _ (mlp-cl/init
           (make-mlp-config max-len field-size conv-size conv-depth)
           seed)
        [in-nd lbl-nd] (make-input-labels field-size max-len)]
    (dosync (ref-set mlp-cl/debug true))
    (loop [i 0, [[inputs labels] & bs] (make-minibatches 16 in-nd lbl-nd)]
      (if (< iter i)
        :done
        (do
          (mlp-cl/run-minibatch inputs labels learning-rate)
          (when (= (mod i 200) 0)
            (printf "i: %6d err: %8.2f\n" i
             (mlp-cl/fw-err-subbatch in-nd lbl-nd))
            (flush))
          (recur (+ i 1) bs))))
    (doseq [m [in-nd lbl-nd]] (mlp-cl/release-mem m)))
  (println "end  : " (.toString (Date.)))
  (mlp-cl/finalize))
