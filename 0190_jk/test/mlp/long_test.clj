(ns mlp.long-test
  (:require [clojure.test :refer :all  ]
            [mlp.mlp-jk   :as    mlp-jk]
            [clojure.pprint            ]))

(deftest ^:long-test comparator
  (mlp-jk/init [;{:type :dense         :size [1 1]}
                {:type :conv :size [1 1 1] :isize [1 1 1] :pad [0 0 0 0]}
                {:type :offset        :size [1  ]}
                {:type :sigmoid       :size [1  ]}
                {:type :cross-entropy :size [1  ]}])
  (let [inputs (mapv float-array
                     [[0.0] [0.4] [0.6] [1.0]])
        labels (mapv float-array
                     [[0  ] [0  ] [1  ] [1  ]])]
    (dotimes [i 501]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 20) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))

(deftest ^:long-test ident-8
  (mlp-jk/init [;{:type :dense         :size [3 4]}
                {:type :conv :size [3 1 4] :isize [3 1 1] :pad [0 0 0 0]}
                {:type :offset        :size [4  ]}
                {:type :sigmoid       :size [4  ]}
                ;{:type :dense         :size [4 5]}
                {:type :conv :size [1 4 5] :isize [1 4 1] :pad [0 0 0 0]}
                {:type :offset        :size [5  ]}
                {:type :sigmoid       :size [5  ]}
                {:type :cross-entropy :size [5  ]}])
  (let [inputs (mapv float-array
                     [[0 0 0]
                      [0 0 1]
                      [0 1 0]
                      [0 1 1]
                      [1 0 0]
                      [1 0 1]
                      [1 1 0]
                      [1 1 1]])
        labels (mapv float-array
                     [[0 0 0 0 1]
                      [0 0 1 0 1]
                      [0 1 0 0 1]
                      [0 1 1 1 1]
                      [1 0 0 0 1]
                      [1 0 1 1 1]
                      [1 1 0 1 1]
                      [1 1 1 1 0]])]
    (dotimes [i 501]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 20) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) i 1))

(deftest ^:long-test ident-64
  (mlp-jk/init [{:type :dense         :size [64 64]}
                {:type :offset        :size [64   ]}
                {:type :sigmoid       :size [64   ]}
                {:type :cross-entropy :size [64   ]}])
  (let [v (map (partial one-hot 64) (range 64))
        inputs (mapv float-array v)
        labels (mapv float-array v)]
    (dotimes [i 1001]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 50) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))

(deftest ^:long-test ident-64-2-layers
  (mlp-jk/init [{:type :dense         :size [64 64]}
                {:type :offset        :size [64   ]}
                {:type :sigmoid       :size [64   ]}
                ;{:type :dense         :size [64 64]}
                {:type :conv :size [4 16 64] :isize [4 16 1] :pad [0 0 0 0]}
                {:type :offset        :size [64 64]}
                {:type :sigmoid       :size [64   ]}
                {:type :cross-entropy :size [64   ]}])
  (let [v (map (partial one-hot 64) (range 64))
        inputs (mapv float-array v)
        labels (mapv float-array v)]
    (dotimes [i 301]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 50) 0)
        (printf "i: %5d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))

(deftest ^:long-test xor
  (mlp-jk/init [{:type :dense         :size [2 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 1]}
                {:type :offset        :size [1  ]}
                {:type :sigmoid       :size [1  ]}
                {:type :cross-entropy :size [1  ]}])
  (let [inputs (mapv float-array
                     [[0 0] [0 1] [1 0] [1 1]])
        labels (mapv float-array
                     [[0] [1] [1] [0]])]
    (dotimes [i 4001]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))

(deftest ^:long-test xor-softmax
  (mlp-jk/init [{:type :dense         :size [2 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 3]}
                {:type :offset        :size [3  ]}
                {:type :sigmoid       :size [3  ]}
                {:type :dense         :size [3 2]}
                {:type :offset        :size [2  ]}
                {:type :softmax       :size [2  ]}
                {:type :cross-entropy :size [2  ]}])
  (let [inputs (mapv float-array
                     [[0 0] [0 1] [1 0] [1 1]])
        labels (mapv float-array
                     [[0 1] [1 0] [1 0] [0 1]])]
    (dotimes [i 4001]
      (mlp-jk/run-minibatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-jk/fw-err-subbatch inputs labels))
        (flush)))))
