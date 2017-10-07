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

(defn decode [strs]
  (mapv (fn [str]
          (mapv (fn [s]
                  (->> (Integer/parseInt s 16)
                       (radix 2)
                       (#(concat % (repeat 0)))
                       (take 3)
                       vec))
                (re-seq #"\S" str)))
        strs))

(deftest test-trace-net
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
    (is (= (smp/trace-net (decode tested) 2 3 1)
          (decode expected)))))
