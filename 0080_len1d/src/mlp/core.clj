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

(defn make-mlp-config [field-size hidden-layers max-len]
  (concat
   (mapcat (fn [[i0 i1]] [{:type :dense   :size [i0 i1]}
                          {:type :offset  :size [i1   ]}
                          {:type :sigmoid :size [i1   ]}])
           (partition 2 1 (concat [field-size] hidden-layers)))
   [{:type :dense         :size [(last hidden-layers) max-len]}
    {:type :offset        :size [max-len                     ]}
    {:type :softmax       :size [max-len                     ]}
    {:type :cross-entropy :size [max-len                     ]}]))

(defn -main
  [field-size max-len iter learning-rate seed & hidden-layers]
  (println "start: " (.toString (Date.)))
  (let [[field-size max-len iter learning-rate seed]
        (mapv read-string [field-size max-len iter learning-rate seed])
        hidden-layers (mapv read-string hidden-layers)
        _ (mlp-cl/init (make-mlp-config field-size hidden-layers max-len) seed)
        [in-nd lbl-nd] (make-input-labels field-size max-len)]
    ;(pr @mlp-cl/mlp-config)
    (loop [i 0, [[inputs labels] & bs] (make-minibatches 16 in-nd lbl-nd)]
      (if (< iter i)
        :done
        (do
          ;(prn inputs)
          ;(mlp-cl/print-matrix (first inputs) 1 5)
          (mlp-cl/run-minibatch inputs labels learning-rate)
          ;(mlp-cl/run-minibatch in-nd lbl-nd learning-rate)
          (when (= (mod i 200) 0)
            (printf "i: %6d err: %8.2f\n" i
             (mlp-cl/fw-err-subbatch in-nd lbl-nd))
            (flush))
          (recur (+ i 1) bs))))
    (doseq [m (concat in-nd lbl-nd)] (CL/clReleaseMemObject m)))
  (println "end  : " (.toString (Date.)))
  (mlp-cl/finalize))
