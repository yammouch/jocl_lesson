(ns mlp.hidden2
  (:gen-class)
  (:require [mlp.mlp-clj :as mlp]
            [clojure.pprint]))

(defn read-file []
  (->> (slurp "data/p001.dat")
       (#(str "[" % "]"))
       read-string
       (#(nth % 4))))

(defn -main [& _]
  (let [layer2 (reduce #(partition %2 %1) (read-file) [4 4 3])]
    (clojure.pprint/pprint layer2)
    (-> [[[1.0 0.0 0.0 0.0]]]
        (mlp/padding 2 2 2 2)
        (mlp/conv-fw (reverse (map reverse layer2)) true)
        clojure.pprint/pprint)))
