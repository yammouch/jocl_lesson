(ns mlp.JKernel-test
  (:require [clojure.test :refer :all]))

(deftest mul-mv-test
  (let [ov (make-array Float/TYPE 3)
        m  (into-array Float/TYPE [1 2 3 4, 2 4 6 8, 3 6 9 12])
        v  (into-array Float/TYPE [4 3 2 1])]
    (JKernel/mul_mv 3 4 ov m v)
    (is (every? #(< -0.01 % 0.01)
                (map - ov [20 40 60])))))

(deftest mul-mv-time
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

(deftest mul-vm-time
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

(deftest mul-vv-time
  (let [cr 4096 cc 4096
        om (make-array Float/TYPE (* cr cc))
        vr (make-array Float/TYPE cr)
        vc (make-array Float/TYPE cc)]
    (println "time for mul_vv")
    (time (JKernel/mul_vv cr cc om vr vc false))))

(deftest sigmoid-fw-test
  (let [n 11
        v  (into-array Float/TYPE (range -5 (+ -5 n)))
        ov (make-array Float/TYPE n)]
    (JKernel/sigmoid_fw n ov v)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (/ 1.0 (+ 1.0 (Math/exp (- %2)))))
                     ov v)))))

(deftest sigmoid-fw-time
  (let [len 4096
        ov (make-array Float/TYPE len)
        v  (make-array Float/TYPE len)]
    (println "time for sigmoid_fw")
    (time (JKernel/sigmoid_fw len ov v))))

(deftest sigmoid-bw-test
  (let [fw-out (into-array Float/TYPE (range 0.1 0.91 0.1))
        back-grad (into-array Float/TYPE
                   (take (count fw-out) (iterate (partial + 0.5) 0.05)))
        n (count fw-out)
        ov (make-array Float/TYPE n)]
    (JKernel/sigmoid_bw n ov fw-out back-grad)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (* %3 %2 (- 1.0 %2)))
                     ov fw-out back-grad)))))

(deftest sigmoid-bw-time
  (let [len 4096
        ov        (make-array Float/TYPE len)
        fw-out    (make-array Float/TYPE len)
        back-grad (make-array Float/TYPE len)]
    (println "time for sigmoid_bw")
    (time (JKernel/sigmoid_bw len ov fw-out back-grad))))
