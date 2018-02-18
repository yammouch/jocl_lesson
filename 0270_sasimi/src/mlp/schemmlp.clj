(ns mlp.schemmlp
  (:require [mlp.util      :as utl]
            [mlp.schemprep :as smp]))

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

(defn slide-policy [p mv]
  (as-> (update-in p [:field     ] smp/slide mv) p
        (update-in p [:cmd :org 0] + (mv 0))
        (update-in p [:cmd :org 1] + (mv 1))
        (update-in p [:cmd :dst  ] +
                     (mv (case (get-in p [:cmd :cmd])
                           :move-y 0 :move-x 1)))))

(defn slide-history [h mv]
  (mapv #(slide-policy % mv) h))
