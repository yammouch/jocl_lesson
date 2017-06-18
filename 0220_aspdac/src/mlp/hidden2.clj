(ns mlp.hidden2
  (:gen-class)
  (:require [mlp.mlp-clj :as mlp]
            [clojure.pprint]))

(defn read-file []
  (let [read-data (->> (slurp "data/p001.dat")
                       (#(str "[" % "]"))
                       read-string)]
    [(reduce #(partition %2 %1) (nth read-data 2) [4 5 3])
     (reduce #(partition %2 %1) (nth read-data 4) [4 4 3])]))

(defn -main [& _]
  (let [[l1 l2] (read-file)
        l12 (-> [[[1.0 0.0 0.0 0.0]]]
                (mlp/padding 2 2 2 2)
                (mlp/conv-fw (reverse (map reverse l2)) true))]
    (clojure.pprint/pprint l2)
    (clojure.pprint/pprint l12)))
