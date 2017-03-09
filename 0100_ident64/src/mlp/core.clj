(ns mlp.core
  (:gen-class))

(import '(java.util Date))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(require 'mlp.mlp-cl)
(alias 'mlp-cl 'mlp.mlp-cl)

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) i 1))

(defn -main [& _]
  (println "start: " (.toString (Date.)))
  (mlp-cl/init [{:type :dense         :size [64 64]}
                {:type :offset        :size [64   ]}
                {:type :sigmoid       :size [64   ]}
                ;{:type :dense         :size [64 64]}
                {:type :conv :size [4 16 64] :isize [4 16 1] :pad [0 0 0 0]}
                {:type :offset        :size [64 64]}
                {:type :sigmoid       :size [64   ]}
                {:type :cross-entropy :size [64   ]}])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (map (partial one-hot 64) (range 64))
        inputs (mlp-cl/pack ctx v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    ;(dotimes [i 1501]
    (dotimes [i 51]
      (mlp-cl/run-minibatch inputs labels)
      ;(when (= (mod i 50) 0)
      (when true
        (printf "i: %5d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)))
    (println "end  : " (.toString (Date.)))
    (doseq [m [inputs labels]] (mlp-cl/release-mem m)))
  (mlp-cl/finalize))
