(ns mlp.long-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-cl   :as    mlp-cl]
            [mlp.cl       :as    cl    ]
            [clojure.pprint            ])
  (:import  [org.jocl CL Sizeof Pointer]))

(deftest ^:long-test ident-8
  (mlp-cl/init [3 4 5])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (map (partial cl/create-buffer ctx :f)
                    [[0 0 0]
                     [0 0 1]
                     [0 1 0]
                     [0 1 1]
                     [1 0 0]
                     [1 0 1]
                     [1 1 0]
                     [1 1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0 0 0 0 1]
                     [0 0 1 0 1]
                     [0 1 0 0 1]
                     [0 1 1 1 1]
                     [1 0 0 0 1]
                     [1 0 1 1 1]
                     [1 1 0 1 1]
                     [1 1 1 1 0]])]
    (dotimes [i 5001]
    ;(dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        ;(mlp-cl/dump :w 0)
        ;(mlp-cl/dump :b 0)
        ;(mlp-cl/dump :w 1)
        ;(mlp-cl/dump :b 1)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) i 1))

(deftest ^:long-test ident-64
  (mlp-cl/init [64 64])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (map (partial one-hot 64) (range 64))
        inputs (mapv (partial cl/create-buffer ctx :f) v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    (dotimes [i 5001]
    ;(dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        ;(mlp-cl/dump :w 0)
        ;(mlp-cl/dump :b 0)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))

(deftest ^:long-test ident-64-2-layers
  (mlp-cl/init [64 64 64])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (map (partial one-hot 64) (range 64))
        inputs (mapv (partial cl/create-buffer ctx :f) v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    (dotimes [i 20001]
    ;(dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 500) 0)
        (printf "i: %5d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        ;(mlp-cl/dump :w 0)
        ;(mlp-cl/dump :b 0)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))

(deftest ^:long-test xor
  (mlp-cl/init [2 3 3 1])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (map (partial cl/create-buffer ctx :f)
                    [[0 0]
                     [0 1]
                     [1 0]
                     [1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0]
                     [1]
                     [1]
                     [0]])]
    (dotimes [i 4001]
    ;(dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        ;(mlp-cl/dump :w 0)
        ;(mlp-cl/dump :b 0)
        ;(mlp-cl/dump :w 1)
        ;(mlp-cl/dump :b 1)
        ;(mlp-cl/dump :w 2)
        ;(mlp-cl/dump :b 2)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
