(ns mlp.schemprep
  (:require [mlp.util :as utl]
            [clojure.pprint]))

(defn format-field [field]
  (mapv (fn [row]
          (as-> row r
                (map #(->> (reverse %)
                           (reduce (fn [acc x] (+ (* acc 2) x)))
                           (format "%02X"))
                     r)
                (interpose " " r)
                (apply str r)))
        field))

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
    (utl/mapd fslide o field)))

(defn slide [field v]
  (reduce (fn [fld [n o]] (slide-1d fld n o))
          field
          (map vector v (range))))
