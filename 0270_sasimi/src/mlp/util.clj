(ns mlp.util
  (:require [clojure.pprint]))

(defn mapd [f d l & ls]
  (if (<= d 0)
    (apply f l ls)
    (apply mapv (partial mapd f (- d 1)) l ls)))

(defn xorshift [x y z w]
  (let [t  (bit-xor x (bit-shift-left x 11))
        wn (bit-and 0xFFFFFFFF
                    (bit-xor w (bit-shift-right w 19)
                             t (bit-shift-right t  8)))]
    (cons w (lazy-seq (xorshift y z w wn)))))

(defn lift [[x & xs] n]
  (cond (not x) n
        (< n x) n
        :else (recur xs (+ 1 n))
        ))

(defn rand-nodup [n lt rs]
  (loop [acc (sorted-set)
         accv []
         [x & xs] (map rem rs (range lt (- lt n) -1))]
    (if x
      (let [lifted (lift (seq acc) x)]
        (recur (conj acc lifted) (conj accv lifted) xs))
      accv)))

(defn select [v ns rs]
  (as-> (rand-nodup (apply + ns) (count v) rs) x
        (map #(v %) x)
        (loop [x x, [n & ns] ns, acc []]
          (if (not n)
            acc
            (recur (drop n x) ns (conj acc (take n x)))
            ))))

(defn one-hot [val len]
  (take len (concat (repeat val 0) [1] (repeat 0))))
