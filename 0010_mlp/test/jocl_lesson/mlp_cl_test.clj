(ns jocl-lesson.mlp-cl-test
  (:require [clojure.test :refer :all]
            [jocl-lesson.mlp-cl :as mlp-cl]
            [jocl-lesson.cl     :as cl    ]
            [clojure.pprint               ])
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
  (let [in  [1 2 3 4]
        out [2 3 4 5]
        n (count in)
        {q :queue} @mlp-cl/cl-env
        {add "add" sub "sub"} @mlp-cl/cl-ker
        [mem-out mem-in]
        (map (partial cl/create-buffer (@mlp-cl/cl-env :context) :f)
             [out in])]
    (cl/callk q add nil [n] :m mem-out :m mem-in)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-out n)
                       [3 5 7 9])))
    (cl/callk q sub nil [n] :m mem-out :m mem-in)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-out n)
                       [2 3 4 5])))))

(deftest dense-fw-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "dense_fw"} @mlp-cl/cl-ker
        w 4, h 3
        in [3 2 1]
        ofs [1 3 5 7]
        m  [ 1  2  3  4
             2  4  6  8
             3  6  9 12]
        [mem-m mem-ofs mem-in mem-out :as mems]
        (map (fn [n]
               (cl/create-buffer ctx :f n))
             [m ofs in w])]
    (cl/callk q k nil [w] :m mem-out :m mem-in :m mem-ofs :m mem-m :i w :i h)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-out w)
                       [11 23 35 47])))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest dense-bw-m-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "dense_bw_m" k-ov "dense_bw_m_ov"} @mlp-cl/cl-ker
        [mem-in mem-out mem-m :as mems]
        (map (partial cl/create-buffer ctx :f)
             [[1 2 3] [1 2 3 4] (repeat 12 1)]
             )]
    (cl/callk q k    nil [3 4] :m mem-m :m mem-in :m mem-out :i 4)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-m 12)
                       [2 3 4 5, 3 5 7 9, 4 7 10 13])))
    (cl/callk q k-ov nil [3 4] :m mem-m :m mem-in :m mem-out :i 4)
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

(deftest dense-bw-v-test
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "dense_bw_v"} @mlp-cl/cl-ker
        [mem-in mem-out mem-m :as mems]
        (map (partial cl/create-buffer ctx :f)
             [3 [4 3 2 1] [1 2 3 4, 2 4 6 8, 3 6 9 12]])]
    (cl/callk q k nil [3] :m mem-in :m mem-out :m mem-m :i 4)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-in 3)
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
        n (count in)
        [mem-in mem-out :as mems]
        (map (partial cl/create-buffer ctx :f) [n in])]
    (cl/callk q k nil [n] :m mem-in :m mem-out)
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (* %2 (- 1.0 %2)))
                     (cl/read-float q mem-in n)
                     in)))
    (cl/callk q k nil [n] :m mem-out :m mem-out) ; overwrite test
    (is (every? #(< -0.01 % 0.01)
                (map #(- %1 (* %2 (- 1.0 %2)))
                     (cl/read-float q mem-out n)
                     in)))
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
