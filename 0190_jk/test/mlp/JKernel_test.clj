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
    (println "time for mul_mv")
    (time (JKernel/mul_mv cr cc ov m v))))

(deftest mul-vm-test
  (let [ov (make-array Float/TYPE 4)
        v  (into-array Float/TYPE [3 2 1])
        m  (into-array Float/TYPE [ 1  2  3  4
                                    2  4  6  8
                                    3  6  9 12])]
    (JKernel/mul_vm 3 4 ov v m)
    (is (every? #(< -0.01 % 0.01)
                (map - ov [10 20 30 40])))))

(deftest mul-vm-test-time
  (let [cr 4096 cc 4096
        ov (make-array Float/TYPE cc)
        v  (make-array Float/TYPE cr)
        m  (make-array Float/TYPE (* cr cc))]
    (println "time for mul_vm")
    (time (JKernel/mul_vm cr cc ov v m))))

(deftest mul-vv-test
  (let [vr (into-array Float/TYPE [1 2 3])
        vc (into-array Float/TYPE [1 2 3 4])
        om (into-array Float/TYPE (repeat 12 1))]
    (JKernel/mul_vv 3 4 om vr vc true)
    (is (every? #(< -0.01 % 0.01)
                (map - om [2 3 4 5, 3 5 7 9, 4 7 10 13])))
    (JKernel/mul_vv 3 4 om vr vc false)
    (is (every? #(< -0.01 % 0.01)
                (map - om [1 2 3 4, 2 4 6 8, 3 6 9 12])
                ))))

(deftest mul-vv-test-time
  (let [cr 4096 cc 4096
        om (make-array Float/TYPE (* cr cc))
        vr (make-array Float/TYPE cr)
        vc (make-array Float/TYPE cc)]
    (println "time for mul_vv")
    (time (JKernel/mul_vv cr cc om vr vc false))))
