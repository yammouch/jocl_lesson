(ns mlp.JKernel-test
  (:require [clojure.test :refer :all]))

(deftest mul-mv-test
  (let [ov (make-array Float/TYPE 3)
        m  (into-array Float/TYPE [1 2 3 4, 2 4 6 8, 3 6 9 12])
        v  (into-array Float/TYPE [4 3 2 1])]
    (JKernel/mul_mv 3 4 ov m v)
    (is (every? #(< -0.01 % 0.01)
                (map - ov [20 40 60])))))

(deftest mul-mv-test-time
  (let [cr 4096 cc 4096
        ov (make-array Float/TYPE cr)
        m  (make-array Float/TYPE (* cr cc))
        v  (make-array Float/TYPE cc)]
    (time (JKernel/mul_mv cr cc ov m v))))
