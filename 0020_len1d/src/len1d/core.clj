; lein run  10 10 20001 30 30 # does not converges

(ns len1d.core
  (:gen-class))

(import '(org.jocl CL))

(require 'len1d.cl)
(alias 'cl 'len1d.cl)

(require 'len1d.mlp-cl)
(alias 'mlp-cl 'len1d.mlp-cl)

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

(defn -main
  [field-size max-len iter & hidden-layers]
  (let [[field-size max-len iter] (mapv read-string [field-size max-len iter])
        hidden-layers (mapv read-string hidden-layers)
        _ (mlp-cl/init (concat [field-size] hidden-layers [max-len]))
        [in-nd lbl-nd] (make-input-labels field-size max-len)]
    (loop [i 0, [[inputs labels] & bs] (make-minibatches 16 in-nd lbl-nd)]
      (if (< iter i)
        :done
        (do
          (mlp-cl/run-subbatch inputs labels)
          (when (= (mod i 200) 0)
            (printf "i: %4d err: %8.2f\n" i
             (mlp-cl/fw-err-subbatch inputs labels))
            (flush))
          (recur (+ i 1) bs))))
    (doseq [m (concat in-nd lbl-nd)] (CL/clReleaseMemObject m))))
