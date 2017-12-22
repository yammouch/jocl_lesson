(ns mlp.parsec)

(defn eos [s]
  (if (empty? s)
    [:eos s]
    [nil s]))

(defn char [pred]
  (fn [s]
    (if (and (not (empty? s))
        (pred (first s)))
      [(first s) (rest s)]
      [nil s])))

(defn all [& ps]
  (fn [s]
    (loop [[p & p2 :as ps] ps, s s, acc []]
      (if (empty? ps)
        [acc s]
        (let [[parsed remain] (p s)]
          (if parsed
            (recur p2 remain (conj acc parsed))
            [nil remain]))))))

(defn oneof [& ps]
  (fn [s]
    (loop [[p & p2 :as ps] ps]
      (if (empty? ps)
        [nil s]
        (let [[parsed remain] (p s)]
          (if parsed
            [parsed remain]
            (recur p2)))))))

(defn times [p lo up]
  (fn [str]
    (loop [i 0, acc [], remain str]
      (if (<= lo i)
        (loop [i lo, acc acc, remain remain]
          (if (and up (<= up i))
            [acc remain]
            (let [[parsed remain] (p remain)]
              (if parsed
                (recur (+ i 1) (conj acc parsed) remain)
                [acc remain]))))
        (let [[parsed remain] (p remain)]
          (if parsed
            (recur (+ i 1) (conj acc parsed) remain)
            [nil remain])))))) 
