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

(defn slide-1d [field n o]
  (let [empty (as-> field x
                    (iterate first x)
                    (take-while coll? x)
                    (map count x)
                    (drop (+ o 1) x)
                    (reverse x)
                    (reduce #(vec (repeat %2 %1)) 0 x))
        fslide (fn [l] (vec (take (count l)
                                  (concat (repeat n empty)
                                          (drop (- n) l)
                                          (repeat empty)))))]
    (mapd fslide o field)))

(defn slide [field v]
  (reduce (fn [fld [n o]] (slide-1d fld n o))
          field
          (map vector v (range))))
