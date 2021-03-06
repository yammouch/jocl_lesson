(ns mlp.JKernel-test
  (:require [clojure.test :refer :all]
            [mlp.mlp-clj :as mlp-clj]))

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

(deftest softmax-test
  (let [v (into-array Float/TYPE [1 2 3 4])
        n (count v)
        ov (make-array Float/TYPE n)]
    (JKernel/softmax (int-array [n]) ov v)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 %2) ov (mlp-clj/softmax v))
                ))))

(deftest softmax-partial-test
  (let [v [[1 2 3] [4 5 6] [7 8]]
        n (map count v)
        ov (make-array Float/TYPE (apply + n))]
    (JKernel/softmax (int-array n) ov (float-array (apply concat v)))
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 %2) ov (mapcat mlp-clj/softmax v))
                ))))

(deftest softmax-time
  (let [len 4096
        ov (make-array Float/TYPE len)
        v  (make-array Float/TYPE len)]
    (println "time for softmax")
    (time (JKernel/softmax (int-array [len]) ov v))))

(deftest quadratic-bw-test
  (let [fw-out (into-array Float/TYPE [0.5 0.5 0.5 0.5])
        expc   (into-array Float/TYPE [0   0   1   1  ])
        n (count fw-out)
        ov (make-array Float/TYPE n)
        learning-rate (float 0.1)]
    (JKernel/quadratic_bw n ov fw-out expc learning-rate)
    (is (every? #(< -0.01 % 0.01)
                (map - ov [0.0125 0.0125 -0.0125 -0.0125])))))

(deftest quadratic-bw-time
  (let [len 4096
        ov     (make-array Float/TYPE len)
        fw-out (make-array Float/TYPE len)
        expc   (make-array Float/TYPE len)]
    (println "time for quadratic_bw")
    (time (JKernel/quadratic_bw len ov fw-out expc (float 0.1)))))

(deftest cross-entropy-bw-test
  (let [fw-out (into-array Float/TYPE [0.5 0.5 0.5 0.5])
        expc   (into-array Float/TYPE [0   0   1   1  ])
        n (count fw-out)
        ov (make-array Float/TYPE n)
        learning-rate (float 0.1)]
    (JKernel/cross_entropy_bw n ov fw-out expc learning-rate)
    (is (every? #(< -0.01 % 0.01)
                (map - ov [0.05 0.05 -0.05 -0.05])))))

(deftest cross-entropy-bw-time
  (let [len 4096
        ov     (make-array Float/TYPE len)
        fw-out (make-array Float/TYPE len)
        expc   (make-array Float/TYPE len)]
    (println "time for cross_entropy_bw")
    (time (JKernel/cross_entropy_bw len ov fw-out expc (float 0.1)))))

(defn test-data-ramp [seed & dim]
  (reduce #(partition %2 %1)
          (map (partial * seed) (range (apply * dim)))
          (butlast dim)))

(defn conv-fw-test1 [ih iw id ch cw cd pu pd pl pr]
  (let [i (test-data-ramp 0.1    id iw ih)
        c (test-data-ramp 0.1 cd id cw ch)
        conved (mlp-clj/conv-fw (mlp-clj/padding i pu pd pl pr) c)
        rh (count conved) rw (count (first conved))
        addend (test-data-ramp 0.2 cd rw rh)
        [mem-result mem-i mem-c]
        (map #(into-array Float/TYPE (flatten %)) [addend i c])]
    (JKernel/conv_fw rh rw ih iw id ch cw cd pu pl mem-result mem-i mem-c)
    (is (every? #(< -0.01 % 0.01) ; 1% of tolerance
                (map (fn [cal ref]
                       (if (< -1.0 ref 1.0)
                         (- cal ref)
                         (- (/ cal ref) 1.0)))
                     mem-result
                     (flatten conved))))))

(deftest conv-fw-test
  (conv-fw-test1  6  6  3  3  3  6  1  1  1  1)
  (conv-fw-test1 12 11 10  9  8  7  6  5  4  3)
  (conv-fw-test1 11 10  9  8  7  6  5  4  3  2))

(deftest conv-fw-time
  (let [ih 20 iw 20 id 20 ch 10 cw 10 cd 20 pu 5 pd 5 pl 5 pr 5
        rh (+ ih (- ch) 1 pu pd)
        rw (+ iw (- cw) 1 pl pr)
        result (make-array Float/TYPE (* rh rw cd))
        input  (make-array Float/TYPE (* ih iw id))
        coeff  (make-array Float/TYPE (* ch cw cd id))]
    (println "time for conv_fw")
    (time
      (JKernel/conv_fw rh rw ih iw id ch cw cd pu pl result input coeff))))

(defn conv-bw-u-test1 [ih iw id ch cw cd pu pd pl pr]
  (let [i (test-data-ramp 0.1 id iw ih)
        c (test-data-ramp 0.1 cd cw ch)
        conved (mlp-clj/conv-fw (mlp-clj/padding i pu pd pl pr) c)
        rh (count conved) rw (count (first conved))
        addend (test-data-ramp 0.2 cd id rw rh)
        result (mlp-clj/+r conved addend)
        [mem-result mem-i mem-c]
        (map #(into-array Float/TYPE (flatten %)) [addend i c])]
    (JKernel/conv_bw_u rh rw ih iw id ch cw cd pu pl false
     mem-result mem-i mem-c)
    (is (every? #(< -0.01 % 0.01) ; 1% of tolerance
                (map (fn [cal ref]
                       (if (< -1.0 ref 1.0)
                         (- cal ref)
                         (- (/ cal ref) 1.0)))
                     mem-result (flatten result))))
    (JKernel/conv_bw_u rh rw ih iw id ch cw cd pu pl true
     mem-result mem-i mem-c)
    (is (every? #(< -0.01 % 0.01) ; 1% of tolerance
                (map (fn [cal ref]
                       (if (< -1.0 ref 1.0)
                         (- cal ref)
                         (- (/ cal ref) 1.0)))
                     mem-result (flatten conved)
                     )))))

(deftest conv-bw-u-test
  (conv-bw-u-test1  6  6  3  3  3  6  1  1  1  1)
  (conv-bw-u-test1 12 11 10  9  8  7  6  5  4  3)
  (conv-bw-u-test1 11 10  9  8  7  6  5  4  3  2))

(deftest conv-bw-u-time
  (let [ih 20 iw 20 id 20 ch 10 cw 10 cd 20 pu 5 pd 5 pl 5 pr 5
        rh (+ ih (- ch) 1 pu pd)
        rw (+ iw (- cw) 1 pl pr)
        result (make-array Float/TYPE (* rh rw id cd))
        input  (make-array Float/TYPE (* ih iw id))
        coeff  (make-array Float/TYPE (* ch cw cd))]
    (println "time for conv_bw_u")
    (time
      (JKernel/conv_bw_u rh rw ih iw id ch cw cd pu pl false
       result input coeff))))

(defn conv-bw-b-test1 [ih iw id ch cw cd pu pd pl pr]
  (let [i (test-data-ramp 0.1 id    iw ih)
        c (test-data-ramp 0.1 id cd cw ch)
        conved (mlp-clj/conv-fw (mlp-clj/padding i pu pd pl pr)
                                (reverse (map reverse c))
                                true)
        rh (count conved) rw (count (first conved))
        addend (test-data-ramp 0.2 cd rw rh)
        [mem-result mem-i mem-c :as mems]
        (map #(into-array Float/TYPE (flatten %)) [addend i c])]
    (JKernel/conv_bw_b rh rw ih iw id ch cw cd pu pl mem-result mem-i mem-c)
    (is (every? #(< -0.01 % 0.01) ; 1% of tolerance
                (map (fn [cal ref]
                       (if (< -0.01 ref 0.01)
                         cal
                         (- (/ cal ref) 1.0)))
                     mem-result (flatten conved)
                     )))))

(deftest conv-bw-b-test
  (conv-bw-b-test1  1  1  2  1  1  1  0  0  0  0)
  (conv-bw-b-test1  6  6  3  3  3  6  1  1  1  1)
  (conv-bw-b-test1 12 11 10  9  8  7  6  5  4  3)
  (conv-bw-b-test1 11 10  9  8  7  6  5  4  3  2))

(deftest conv-bw-b-time
  (let [ih 20 iw 20 id 20 ch 10 cw 10 cd 20 pu 5 pd 5 pl 5 pr 5
        rh (+ ih (- ch) 1 pu pd)
        rw (+ iw (- cw) 1 pl pr)
        result (make-array Float/TYPE (* rh rw cd))
        input  (make-array Float/TYPE (* ih iw id))
        coeff  (make-array Float/TYPE (* ch cw cd id))]
    (println "time for conv_bw_b")
    (time
      (JKernel/conv_bw_b rh rw ih iw id ch cw cd pu pl
       result input coeff))))
