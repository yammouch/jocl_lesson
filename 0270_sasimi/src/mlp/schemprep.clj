(ns mlp.schemprep
  (:require [clojure.pprint]))

(defn mapd [f d l & ls]
  (if (<= d 0)
    (apply f l ls)
    (apply mapv (partial mapd f (- d 1)) l ls)))

(defn count-empty-row-up [field]
  (as-> field fld
        (take-while (partial every? (partial every? zero?)) fld)
        (count fld)))

(defn room [field]
  (let [b (reverse field)
        l (apply map vector field)
        r (reverse l)]
    (map count-empty-row-up [field b l r])))

(defn slide [field n o]
  (let [empty (as-> field x
                    (iterate first x)
                    (take-while coll? x)
                    (map count x)
                    (drop (+ o 1) x)
                    (reverse x)
                    (reduce #(vec (repeat %2 %1)) 0 x))
        fslide (if (< n 0)
                 (fn [l] (vec (take (count l)
                                    (concat (drop (- n) l)
                                            (repeat empty)))))
                 (fn [l] (vec (take (count l)
                                    (concat (repeat n empty)
                                            l)))))]
    (mapd fslide o field)))
