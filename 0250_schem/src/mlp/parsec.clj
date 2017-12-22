(ns mlp.parsec)

(defn char [pred]
  (fn [str]
    (if (and (not (empty? str))
        (pred (first str)))
      [(first str) (rest str)]
      [nil str])))

(defn all [& ps]
  (fn [str]
    (loop [[p & pr :as ps] ps, s str, acc []]
      (if (empty? ps)
        [acc s]
        (let [[parsed remain] (p s)]
          (if parsed
            (recur pr remain (conj acc parsed))
            [nil remain]))))))

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
