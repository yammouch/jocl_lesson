(ns mlp.schemanip-test
  (:require [clojure.test :refer :all]
            [mlp.schemanip :as smp]
            [clojure.pprint]))

(deftest test-surrouding
  (is (= (smp/surrounding 4 6)
         [[3 6 0 3 6 0]
          [4 6 0 5 6 0]
          [4 5 1 4 5 1]
          [4 6 1 4 7 1]])))

(defn radix [rdx i]
  (loop [i i acc []]
    (if (<= i 0)
      acc
      (recur (quot i rdx)
             (conj acc (rem i rdx))
             ))))

(defn decode1 [n str]
  (mapv (fn [s]
          (->> (Integer/parseInt s 16)
               (radix 2)
               (#(concat % (repeat 0)))
               (take n)
               vec))
        (re-seq #"\S" str)))

(defn decode [n strs] (mapv (partial decode1 n) strs))

(defn mapd [d f s & ss]
  (if (<= d 0)
    (apply f s ss)
    (apply mapv (partial mapd (- d 1) f) s ss)))

(deftest test-trace
  (let [test-pattern
        ["0000100000" "0000000000"
         "0000100000" "0000000000"
         "0022322200" "0022222200"
         "0000100000" "0000000000"
         "0000100000" "0000000000"
         "0000100000" "0000000000"
         "0022722200" "0000000000"
         "0000100000" "0000000000"
         "0000100000" "0000000000"
         "0000100000" "0000000000"]
        [tested expected] (apply map vector (partition 2 test-pattern))]
    (is (= (smp/trace (decode 3 tested) 2 3 1)
           (decode 2 expected)))))

(deftest test-beam
  (let [test-pattern
        ["0000000000"
         "0032227200"
         "0010001010"
         "0010001010"
         "0000000000"
         "0000000000"]]
    (is (= (smp/beam (decode 3 test-pattern) [1 4] 1)
           [[1 2] [1 6]]))
    (is (= (smp/beam (decode 3 test-pattern) [1 2] 0)
           [[1 2] [4 2]]))))

(deftest test-drawable?
  (let [test-pattern
        ["0000000000" "0000000000"
         "0032227200" "0032227200"
         "0010001010" "0010001010"
         "0010001010" "0010001010"
         "0000000000" "0000000000"
         "0032227200" "0000000000"
         "0010001010" "0000000000"
         "0010001010" "0000000000"
         "0000000000" "0000000000"]
        [field traced] (->> test-pattern
                            (map (partial decode1 3))
                            (partition 2)
                            (apply map vector))]
    (is      (smp/drawable? 4 2 1 traced field) )
    (is      (smp/drawable? 6 2 1 traced field) )
    (is (not (smp/drawable? 5 3 1 traced field)))
    ))

(deftest test-add-dot
  (let [test-pattern
        ["0000000000" "0000000000" "0000000000"
         "0000000000" "0000000000" "0000000000"
         "0022232220" "0022272220" "0022272220"
         "0000010000" "0000010000" "0000010000"
         "0002210000" "0002210000" "0002250000"
         "0000010000" "0000010000" "0000010000"
         "0032232200" "0032232200" "0032232200"
         "0010010000" "0010010000" "0010010000"
         "0022200000" "0022200000" "0022200000"
         "0000000000" "0000000000" "0000000000"]
        [field ex1 ex2] (->> test-pattern
                             (map (partial decode1 3))
                             (partition 3)
                             (apply map vector))]
    (is (= (smp/add-dot [2 2] 8 1 field field) ex1  ))
    (is (= (smp/add-dot [2 5] 8 0 field field) ex2  ))
    ;(clojure.pprint/pprint
    ; (mapd 2 (comp (partial reduce (fn [acc x] (+ (* acc 2) x)))
    ;               reverse)
    ;         (smp/add-dot [2 5] 8 0 field field)))
    (is (= (smp/add-dot [7 2] 8 1 field field) field))
    ))

(deftest test-draw-net-1
  (let [test-pattern 
        ["0000000000" "0000000000" "0000000000"
         "0000000000" "0022222200" "0000000000"
         "0000000000" "0000000000" "0000100000"
         "0000000000" "0000000000" "0000100000"
         "0000000000" "0000000000" "0000100000"
         "0000000000" "0000000000" "0000100000"
         "0000000000" "0000000000" "0000100000"
         "0000000000" "0000000000" "0000000000"
         "0000000000" "0000000000" "0000000000"
         "0000000000" "0000000000" "0000000000"]
        [field ex1 ex2] (->> test-pattern
                             (map (partial decode1 3))
                             (partition 3)
                             (apply map vector))]
    (is (= (smp/draw-net-1 [1 2] 8 1 field) ex1))
    (is (= (smp/draw-net-1 [2 4] 7 0 field) ex2))))