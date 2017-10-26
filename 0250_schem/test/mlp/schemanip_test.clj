(ns mlp.schemanip-test
  (:require [clojure.test :refer :all]
            [mlp.schemanip :as smp]))

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

(deftest test-beam-h
  (let [test-pattern
        ["0000000000"
         "0032227200"
         "0010001010"
         "0010001010"
         "0000000000"
         "0000000000"]]
    (is (= (smp/beam-h (decode 3 test-pattern) 1 4)
           [2 6]))))

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
