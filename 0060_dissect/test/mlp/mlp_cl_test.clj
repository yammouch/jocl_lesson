(ns mlp.mlp-cl-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-cl   :as    mlp-cl]
            [mlp.cl       :as    cl    ]
            [clojure.pprint            ])
  (:import  [org.jocl CL Sizeof Pointer]))

(use-fixtures :once
  (fn [f]
    (mlp-cl/init [1 1])
    (f)
    (mlp-cl/finalize)))

(deftest set0-test
  (let [n 4
        {q :queue} @mlp-cl/cl-env
        {set0 "set0"} @mlp-cl/cl-ker
        mem (cl/create-buffer (@mlp-cl/cl-env :context) :f n)]
    (cl/callk q set0 nil [n] :m mem)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem n) (repeat n 0))))
    (CL/clReleaseMemObject mem)))

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

(deftest dense-bw-ofs-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "dense_bw_ofs" k-ov "dense_bw_ofs_ov"} @mlp-cl/cl-ker
        out [1 2 3 4]
        ofs [1 1 2 2]
        n (count out)
        [mem-ofs mem-out :as mems]
        (map (partial cl/create-buffer ctx :f) [ofs out])]
    (cl/callk q k    nil [n] :m mem-ofs :m mem-out)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-ofs n)
                       [2 3 5 6])))
    (cl/callk q k-ov nil [n] :m mem-ofs :m mem-out)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-ofs n)
                       [1 2 3 4])))
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
        in (range -5 (+ -5 n))
        [mem-in mem-out :as mems]
        (map (partial cl/create-buffer ctx :f) [in n])]
    (cl/callk q k nil [n] :m mem-out :m mem-in)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (/ 1.0 (+ 1.0 (Math/exp (- %2)))))
                     (cl/read-float q mem-out n)
                     in)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest sigmoid-bw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "sigmoid_bw"} @mlp-cl/cl-ker
        in (range 0.1 0.91 0.1)
        out-prop (take (count in) (iterate (partial + 0.5) 0.05))
        n (count in)
        [mem-in mem-out mem-out-prop :as mems]
        (map (partial cl/create-buffer ctx :f) [n in out-prop])]
    (cl/callk q k nil [n] :m mem-in :m mem-out :m mem-out-prop)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (* %3 %2 (- 1.0 %2)))
                     (cl/read-float q mem-in n)
                     in out-prop)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest softmax-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k1 "softmax_fw_step1" k2 "softmax_fw_step2" k3 "softmax_fw_step3"}
        @mlp-cl/cl-ker
        in [1 2 3 4]
        n (count in)
        [mem-in mem-out :as mems]
        (map (partial cl/create-buffer ctx :f) [in (inc n)])]
    (cl/callk q k1 nil [n] :m mem-out :m mem-in)
    (cl/callk q k2 nil [1] :m mem-out :i n)
    (cl/callk q k3 nil [n] :m mem-out :i n)
    (let [exp-in (map #(Math/exp %) in)
          sum (apply + exp-in)]
      (is (every? #(< -0.01 % 0.01)
                  (map #(- %1 (/ %2 sum))
                       (cl/read-float q mem-out n)
                       exp-in))))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest quadratic-bw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "quadratic_bw"} @mlp-cl/cl-ker
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
