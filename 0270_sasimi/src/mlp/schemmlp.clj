(ns mlp.schemmlp
  (:require [mlp.util :as utl]))

(defn mlp-input-cmd [{cmd :cmd [y x] :org dst :dst} [cy cx]]
  (concat (case cmd :move-y [1 0] :move-x [0 1])
          (utl/one-hot y cy)
          (utl/one-hot x cx)
          (utl/one-hot dst (max cy cx))))

(defn make-input-label [pair h w]
  (as-> pair p
        (update-in p [:field] (comp float-array
                                    (partial apply concat)
                                    (partial apply concat)))
        (update-in p [:cmd] (comp float-array #(mlp-input-cmd % [h w])))))
                                  
(defn make-minibatches [sb-size v]
  (map (partial mapv v)
       (partition sb-size (map #(mod % (count v))
                               (utl/xorshift 2 4 6 8)
                               ))))
