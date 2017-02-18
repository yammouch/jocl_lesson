(ns mlp.core
  (:gen-class))

(import '(org.jocl CL))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(require 'mlp.mlp-cl)
(alias 'mlp-cl 'mlp.mlp-cl)

(def inputs-v
  [[0 0 0]
   [0 0 1]
   [0 1 0]
   [0 1 1]
   [1 0 0]
   [1 0 1]
   [1 1 0]
   [1 1 1]])

(def labels-v
  [[0 0 0 0 1]
   [0 0 1 0 1]
   [0 1 0 0 1]
   [0 1 1 1 1]
   [1 0 0 0 1]
   [1 0 1 1 1]
   [1 1 0 1 1]
   [1 1 1 1 0]])

(def mlp-config
  [{:type :dense         :size [3 4]}
   {:type :offset        :size [4  ]}
   {:type :sigmoid       :size [4  ]}
   {:type :dense         :size [4 5]}
   {:type :offset        :size [5  ]}
   {:type :sigmoid       :size [5  ]}
   {:type :cross-entropy :size [5  ]}
   ])

(defn -main [& args]
  (mlp-cl/init mlp-config)
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        inputs (map (partial cl/create-buffer ctx :f) inputs-v)
        labels (map (partial cl/create-buffer ctx :f) labels-v)]
    (dotimes [i 1001]
    ;(dotimes [i 1]
      ;(mlp-cl/dump 0 :p)
      ;(mlp-cl/dump 1 :p)
      ;(mlp-cl/dump 3 :p)
      ;(mlp-cl/dump 4 :p)
      (mlp-cl/run-minibatch inputs labels)
      (when (= (mod i 50) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (mlp-cl/dump 0 :p)
        (mlp-cl/dump 1 :p)
        (mlp-cl/dump 3 :p)
        (mlp-cl/dump 4 :p)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
