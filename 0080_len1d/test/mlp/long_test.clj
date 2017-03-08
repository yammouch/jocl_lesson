(ns mlp.long-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-cl   :as    mlp-cl]
            [mlp.cl       :as    cl    ]
            [clojure.pprint            ])
  (:import  [org.jocl CL Sizeof Pointer]))

(deftest ^:long-test comparator
  (mlp-cl/init [;{:type :dense         :size [1 1]}
                {:type :conv :size [1 1 1] :isize [1 1 1] :pad [0 0 0 0]}
                {:type :offset        :size [1  ]}
                {:type :sigmoid       :size [1  ]}
                {:type :cross-entropy :size [1  ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (mlp-cl/pack ctx [[0.0] [0.4] [0.6] [1.0]] 1)
        labels (map (partial cl/create-buffer ctx :f)
                    [[0  ] [0  ] [1  ] [1  ]])]
    (dotimes [i 501]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 20) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))

(deftest ^:long-test ident-8
  (mlp-cl/init [;{:type :dense         :size [3 4]}
                {:type :conv :size [1 3 4] :isize [1 3 1] :pad [0 0 0 0]}
                {:type :offset        :size [4  ]}
                {:type :sigmoid       :size [4  ]}
                {:type :dense         :size [4 5]}
                {:type :offset        :size [5  ]}
                {:type :sigmoid       :size [5  ]}
                {:type :cross-entropy :size [5  ]}])
  (dosync (ref-set mlp-cl/debug true))
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (mlp-cl/pack ctx [[0 0 0]
                                 [0 0 1]
                                 [0 1 0]
                                 [0 1 1]
                                 [1 0 0]
                                 [1 0 1]
                                 [1 1 0]
                                 [1 1 1]]
                                3)
        labels (map (partial cl/create-buffer ctx :f)
                    [[0 0 0 0 1]
                     [0 0 1 0 1]
                     [0 1 0 0 1]
                     [0 1 1 1 1]
                     [1 0 0 0 1]
                     [1 0 1 1 1]
                     [1 1 0 1 1]
                     [1 1 1 1 0]])]
    ;(dotimes [i 501]
    (dotimes [i 1]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 20) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) i 1))

(deftest ^:long-test ident-64
  (mlp-cl/init [{:type :dense         :size [64 64]}
                {:type :offset        :size [64   ]}
                {:type :sigmoid       :size [64   ]}
                {:type :cross-entropy :size [64   ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (map (partial one-hot 64) (range 64))
        inputs (mlp-cl/pack ctx v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    (dotimes [i 1001]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 50) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))

(deftest ^:long-test ident-64-2-layers
  (mlp-cl/init [{:type :dense         :size [64 64]}
                {:type :offset        :size [64   ]}
                {:type :sigmoid       :size [64   ]}
                {:type :dense         :size [64 64]}
                {:type :offset        :size [64 64]}
                {:type :sigmoid       :size [64   ]}
                {:type :cross-entropy :size [64   ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (map (partial one-hot 64) (range 64))
        inputs (mlp-cl/pack ctx v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    (dotimes [i 1501]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 50) 0)
        (printf "i: %5d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))

(deftest ^:long-test xor
  (mlp-cl/init [{:type :dense         :size [2 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 1]}
                {:type :offset        :size [1  ]}
                {:type :sigmoid       :size [1  ]}
                {:type :cross-entropy :size [1  ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (mlp-cl/pack ctx [[0 0] [0 1] [1 0] [1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0] [1] [1] [0]])]
    (dotimes [i 4001]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))

(deftest ^:long-test xor-softmax
  (mlp-cl/init [{:type :dense         :size [2 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 2]}
                {:type :offset        :size [2  ]}
                {:type :softmax       :size [2  ]}
                {:type :cross-entropy :size [2  ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (mlp-cl/pack ctx [[0 0] [0 1] [1 0] [1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0 1] [1 0] [1 0] [0 1]])]
    (dotimes [i 4001]
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)
        ))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))
