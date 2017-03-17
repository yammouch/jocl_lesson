(ns clj.core
  (:gen-class))

(set! *warn-on-reflection* true)

(defn clojure-like [n]
  (let [l0 (take n (iterate inc 0))
        l1 (take n (iterate inc 1))
        result (map + l0 l1)
        expc (take n (iterate (partial + 2) 1))]
    (time (println (every? #(< -0.01 % 0.01)
                           (map - result expc)
                           )))))

(defn use-array-init [n a0 a1]
  (loop [i 0]
    (if (<= n i)
      :done
      (do (aset-float a0 i i)
          (aset-float a1 i (unchecked-add i 1))
          (recur (unchecked-add i 1))
          ))))

(defn use-array-add [n result ^floats a0 ^floats a1]
  (loop [i 0]
    (if (<= n i)
      :done
      (do (aset-float result i
           (unchecked-add (aget a0 i) (aget a1 i)))
          (recur (unchecked-add i 1))
          ))))

(defn use-array-compare [n ^floats result]
  (loop [i 0 ok? true]
    (if (<= n i)
      ok?
      (if (== (aget result i)
              (unchecked-add (bit-shift-left i 1) 1))
        (recur (unchecked-add i 1) true)
        false))))

(defn use-array [n]
  (let [a0 (time (float-array n))
        a1 (time (float-array n))
        result (time (float-array n))]
    (time          (use-array-init    n        a0 a1) )
    (time          (use-array-add     n result a0 a1) )
    (time (println (use-array-compare n result      )))
    ))

(defn -main
  "I don't do a whole lot ... yet."
  [& _]
  (let [n (bit-shift-left 1 24)]
    (clojure-like n)
    (use-array n)
    ))
