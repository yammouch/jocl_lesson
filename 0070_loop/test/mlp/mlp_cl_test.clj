(ns mlp.mlp-cl-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-cl   :as    mlp-cl]
            [mlp.cl       :as    cl    ]
            [clojure.pprint            ])
  (:import  [org.jocl CL Sizeof Pointer]))

(use-fixtures :once
  (fn [f]
    ;(mlp-cl/init [1 1])
    (mlp-cl/init nil)
    (f)
    (mlp-cl/finalize)))

(deftest add-test
  (let [v0 [1 2 3 4]
        v1 [2 3 4 5]
        n (count v0)
        {q :queue ctx :context} @mlp-cl/cl-env
        {add "add" sub "sub"} @mlp-cl/cl-ker
        [mem-result mem-v0 mem-v1]
        (map (partial cl/create-buffer ctx :f) [n v0 v1])]
    (cl/callk q add nil [n] :m mem-result :m mem-v0 :m mem-v1)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result n)
                       [3 5 7 9])))
    (cl/callk q sub nil [n] :m mem-result :m mem-v0 :m mem-v1)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result n)
                       [-1 -1 -1 -1])))))

(deftest mul-vm-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "mul_vm"} @mlp-cl/cl-ker
        w 4, h 3
        v [3 2 1]
        m [ 1  2  3  4
            2  4  6  8
            3  6  9 12]
        [mem-v mem-m mem-prod :as mems]
        (map (partial cl/create-buffer ctx :f) [v m w])]
    (cl/callk q k nil [w] :m mem-prod :m mem-v :m mem-m :i h :i w)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-prod w)
                       [10 20 30 40])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest mul-vv-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {acc "mul_vv_acc" ov "mul_vv"} @mlp-cl/cl-ker
        [mem-v1 mem-v2 mem-m :as mems]
        (map (partial cl/create-buffer ctx :f)
             [[1 2 3] [1 2 3 4] (repeat 12 1)]
             )]
    (cl/callk q acc nil [3 4] :m mem-m :m mem-v1 :m mem-v2 :i 4)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-m 12)
                       [2 3 4 5, 3 5 7 9, 4 7 10 13])))
    (cl/callk q ov  nil [3 4] :m mem-m :m mem-v1 :m mem-v2 :i 4)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-m 12)
                       [1 2 3 4, 2 4 6 8, 3 6 9 12])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest mul-mv-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "mul_mv"} @mlp-cl/cl-ker
        [mem-prod mem-m mem-v :as mems]
        (map (partial cl/create-buffer ctx :f)
             [3 [1 2 3 4, 2 4 6 8, 3 6 9 12] [4 3 2 1]])]
    (cl/callk q k nil [3] :m mem-prod :m mem-m :m mem-v :i 4)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-prod 3)
                       [20 40 60])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest sigmoid-fw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "sigmoid_fw"} @mlp-cl/cl-ker
        n 11
        v (range -5 (+ -5 n))
        [mem-result mem-v :as mems]
        (map (partial cl/create-buffer ctx :f) [n v])]
    (cl/callk q k nil [n] :m mem-result :m mem-v)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (/ 1.0 (+ 1.0 (Math/exp (- %2)))))
                     (cl/read-float q mem-result n)
                     v)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest sigmoid-bw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "sigmoid_bw"} @mlp-cl/cl-ker
        fw-out (range 0.1 0.91 0.1)
        grad (take (count fw-out) (iterate (partial + 0.5) 0.05))
        n (count fw-out)
        [mem-result mem-fw-out mem-grad :as mems]
        (map (partial cl/create-buffer ctx :f) [n fw-out grad])]
    (cl/callk q k nil [n] :m mem-result :m mem-fw-out :m mem-grad)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (* %3 %2 (- 1.0 %2)))
                     (cl/read-float q mem-result n)
                     fw-out grad)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest softmax-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k1 "softmax_fw_step1" k2 "softmax_fw_step2" k3 "softmax_fw_step3"}
        @mlp-cl/cl-ker
        v [1 2 3 4]
        n (count v)
        [mem-result mem-v :as mems]
        (map (partial cl/create-buffer ctx :f) [v (inc n)])]
    (cl/callk q k1 nil [n] :m mem-result :m mem-v)
    (cl/callk q k2 nil [1] :m mem-result :i n)
    (cl/callk q k3 nil [n] :m mem-result :i n)
    (let [exp-v (map #(Math/exp %) v)
          sum (apply + exp-v)]
      (is (every? #(< -0.01 % 0.01)
                  (map #(- %1 (/ %2 sum))
                       (cl/read-float q mem-result n)
                       exp-v))))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest quadratic-bw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "quadratic_bw"} @mlp-cl/cl-ker
        fw-out [0.5 0.5 0.5 0.5]
        expc   [0   0   1   1  ]
        n (count fw-out)
        learning-rate 0.1
        [mem-fw-out mem-expc mem-result :as mems]
        (map (partial cl/create-buffer ctx :f)
             [fw-out expc n])]
    (cl/callk q k nil [n] :m mem-result :m mem-fw-out :m mem-expc
     :f learning-rate)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result n)
                       [0.0125 0.0125 -0.0125 -0.0125])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest cross-entropy-bw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "cross_entropy_bw"} @mlp-cl/cl-ker
        out  [0.5 0.5 0.5 0.5]
        expc [0   0   1   1  ]
        n (count out)
        learning-rate 0.1
        [mem-out mem-expc mem-in :as mems]
        (map (partial cl/create-buffer ctx :f)
             [out expc n])]
    (cl/callk q k nil [n] :m mem-in :m mem-out :m mem-expc :f learning-rate)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-in n)
                       [0.05 0.05 -0.05 -0.05])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

;(defn formatv [v]
;  (apply str (interpose " " (map (partial format "%6.2f")
;                                 v)))) 
(defn formatv [v]
  (apply str (map (partial format ",%.2f")
             v)))

(defn print-matrix [m]
  (doseq [s (map formatv m)] (println s)))

(defn conv-fw [i c]
  (let [hc (count c)
        wc (count (first c))]
    (map (fn [rows]
           (apply map (fn [& vs]
                        (apply + (map * (apply concat vs) (apply concat c))))
                      (map (partial partition wc 1) rows)))
         (partition hc 1 i))))

(deftest conv-fw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv_fw"} @mlp-cl/cl-ker
        hi 5 wi 6 hc 3 wc 2
        i (partition wi (map (partial * 0.1) (range (* hi wi))))
        c (partition wc (map (partial * 0.1) (range (* hc wc))))
        result (conv-fw i c)
        hr (count result) wr (count (first result))
        [mem-result mem-i mem-c :as mems]
        (map (partial cl/create-buffer ctx :f)
             [(* hr wr) (apply concat i) (apply concat c)])]
    ;(print-matrix i)
    ;(print-matrix c)
    ;(print-matrix result)
    (cl/callk q k nil [wr hr] :m mem-result :m mem-i :m mem-c
     :i wr :i wi :i wc :i hc)
    ;(mlp-cl/print-matrix mem-i      hi wi)
    ;(mlp-cl/print-matrix mem-c      hc wc)
    ;(mlp-cl/print-matrix mem-result hr wr)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* hr wr))
                       (apply concat result))))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest conv-bw-u-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv_bw_u"} @mlp-cl/cl-ker
        hi 5 wi 6 hg 3 wg 2
        i (partition wi (map (partial * 0.1) (range (* hi wi))))
        g (partition wg (map (partial * 0.1) (range (* hg wg))))
        result (conv-fw i g)
        hr (count result) wr (count (first result))
        [mem-result mem-i mem-g :as mems]
        (map (partial cl/create-buffer ctx :f)
             [(* hr wr) (apply concat i) (apply concat g)])]
    ;(print-matrix i)
    ;(print-matrix c)
    ;(print-matrix result)
    (cl/callk q k nil [wr hr] :m mem-result :m mem-i :m mem-g
     :i wr :i wi :i wg :i hg)
    ;(mlp-cl/print-matrix mem-i      hi wi)
    ;(mlp-cl/print-matrix mem-c      hc wc)
    ;(mlp-cl/print-matrix mem-result hr wr)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* hr wr))
                       (apply concat result))))
    (doseq [m mems] (CL/clReleaseMemObject m))))
