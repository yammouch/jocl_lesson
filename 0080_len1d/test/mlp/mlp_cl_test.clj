(ns mlp.mlp-cl-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-cl   :as    mlp-cl]
            [mlp.cl       :as    cl    ]
            [clojure.pprint            ])
  (:import  [org.jocl CL Sizeof Pointer]))

(use-fixtures :once
  (fn [f]
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
        [mem-result mem-v mem-b :as mems]
        (map (partial cl/create-buffer ctx :f) [n v (+ 1 n)])]
    (cl/callk q k1 nil [n] :m mem-b :m mem-v)
    (cl/callk q k2 nil [1] :m mem-b :i n)
    (cl/callk q k3 nil [n] :m mem-result :m mem-b :i n)
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

(defn conv [i c]
  (let [ch (count c)
        cw (count (first c))]
    (map (fn [rows]
           (apply map (fn [& vs]
                        (apply + (map * (apply concat vs) (apply concat c))))
                      (map (partial partition cw 1) rows)))
         (partition ch 1 i))))

(defn padding [m pu pd pl pr]
  (let [w (+ pl pr (count (first m)))]
    (concat (repeat pu (repeat w 0.0))
            (map #(concat (repeat pl 0.0) % (repeat pr 0.0)) m)
            (repeat pd (repeat w 0.0)))))

(defn conv-test1 [ih iw ch cw pu pd pl pr]
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv"} @mlp-cl/cl-ker
        ih 5 iw 6 ch 3 cw 2
        i (partition iw (map (partial * 0.1) (range (* ih iw))))
        c (partition cw (map (partial * 0.1) (range (* ch cw))))
        result (conv (padding i pu pd pl pr) c)
        rh (count result) rw (count (first result))
        [mem-result mem-i mem-c :as mems]
        (map (partial cl/create-buffer ctx :f)
             [(* rh rw) (apply concat i) (apply concat c)])]
    (cl/callk q k nil [rw rh] :m mem-result :m mem-i :m mem-c
     :i rw :i ih :i iw :i ch :i cw :i pu :i pl)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* rh rw))
                       (apply concat result))))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest conv-test
  (conv-test1 5 6 3 2 0 0 0 0)
  (conv-test1 1 3 1 1 0 0 0 0)
  (conv-test1 5 6 3 2 1 0 0 0)
  (conv-test1 5 6 3 2 0 1 0 0)
  (conv-test1 5 6 3 2 0 0 1 0)
  (conv-test1 5 6 3 2 0 0 0 1)
  (conv-test1 5 6 3 2 1 2 3 4)
  (conv-test1 5 6 3 2 2 2 2 2))

(defn conv-acc-test1 [ih iw ch cw pu pd pl pr]
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv_acc"} @mlp-cl/cl-ker
        ih 5 iw 6 ch 3 cw 2
        i (partition iw (map (partial * 0.1) (range (* ih iw))))
        c (partition cw (map (partial * 0.1) (range (* ch cw))))
        conved (conv (padding i pu pd pl pr) c)
        rh (count conved) rw (count (first conved))
        addend (map (partial * 0.2) (range (* rh rw)))
        result (map + (apply concat conved) addend)
        [mem-result mem-i mem-c :as mems]
        (map (partial cl/create-buffer ctx :f)
             [addend (apply concat i) (apply concat c)])]
    (cl/callk q k nil [rw rh] :m mem-result :m mem-i :m mem-c
     :i rw :i ih :i iw :i ch :i cw :i pu :i pl)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* rh rw))
                       result)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest conv-acc-test
  (conv-acc-test1 5 6 3 2 0 0 0 0)
  (conv-acc-test1 5 6 3 2 1 0 0 0)
  (conv-acc-test1 5 6 3 2 0 1 0 0)
  (conv-acc-test1 5 6 3 2 0 0 1 0)
  (conv-acc-test1 5 6 3 2 0 0 0 1)
  (conv-acc-test1 5 6 3 2 1 2 3 4)
  (conv-acc-test1 5 6 3 2 2 2 2 2))

(defn conv-t-test1 [ih iw ch cw pu pd pl pr]
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv_t"} @mlp-cl/cl-ker
        ih 5 iw 6 ch 3 cw 2
        i (partition iw (map (partial * 0.1) (range (* ih iw))))
        c (partition cw (map (partial * 0.1) (range (* ch cw))))
        result (conv (padding i pu pd pl pr) (reverse (map reverse c)))
        rh (count result) rw (count (first result))
        [mem-result mem-i mem-c :as mems]
        (map (partial cl/create-buffer ctx :f)
             [(* rh rw) (apply concat i) (apply concat c)])]
    (cl/callk q k nil [rw rh] :m mem-result :m mem-i :m mem-c
     :i rw :i ih :i iw :i ch :i cw :i pu :i pl)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* rh rw))
                       (apply concat result))))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest conv-t-test
  (conv-t-test1 5 6 3 2 0 0 0 0)
  (conv-t-test1 5 6 3 2 1 0 0 0)
  (conv-t-test1 5 6 3 2 0 1 0 0)
  (conv-t-test1 5 6 3 2 0 0 1 0)
  (conv-t-test1 5 6 3 2 0 0 0 1)
  (conv-t-test1 5 6 3 2 1 2 3 4)
  (conv-t-test1 5 6 3 2 2 2 2 2))

(defn conv-t-acc-test1 [ih iw ch cw pu pd pl pr]
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {k "conv_t_acc"} @mlp-cl/cl-ker
        ih 5 iw 6 ch 3 cw 2
        i (partition iw (map (partial * 0.1) (range (* ih iw))))
        c (partition cw (map (partial * 0.1) (range (* ch cw))))
        conved (conv (padding i pu pd pl pr) (reverse (map reverse c)))
        rh (count conved) rw (count (first conved))
        addend (map (partial * 0.2) (range (* rh rw)))
        result (map + (apply concat conved) addend)
        [mem-result mem-i mem-c :as mems]
        (map (partial cl/create-buffer ctx :f)
             [addend (apply concat i) (apply concat c)])]
    (cl/callk q k nil [rw rh] :m mem-result :m mem-i :m mem-c
     :i rw :i ih :i iw :i ch :i cw :i pu :i pl)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-result (* rh rw))
                       result)))
    (doseq [m mems] (CL/clReleaseMemObject m))))

(deftest conv-t-acc-test
  (conv-t-acc-test1 5 6 3 2 0 0 0 0)
  (conv-t-acc-test1 5 6 3 2 1 0 0 0)
  (conv-t-acc-test1 5 6 3 2 0 1 0 0)
  (conv-t-acc-test1 5 6 3 2 0 0 1 0)
  (conv-t-acc-test1 5 6 3 2 0 0 0 1)
  (conv-t-acc-test1 5 6 3 2 1 2 3 4)
  (conv-t-acc-test1 5 6 3 2 2 2 2 2))
