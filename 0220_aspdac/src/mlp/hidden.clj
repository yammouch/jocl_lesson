(ns mlp.hidden2
  (:gen-class)
  (:require [mlp.mlp-clj :as mlp]
            [clojure.pprint]))

(defn read-file []
  (let [read-data (->> (slurp "data/p001.dat")
                       (#(str "[" % "]"))
                       read-string)]
    [(nth read-data 2) (nth read-data 4)]))

(defn show-l2 [l]
  (->> l
       (partition 20)
       (apply map vector)
       (

(defn -main [& _]
  (let [[l2 l4] (read-file)]
    (show-l2 l2)))
;
;    (clojure.pprint/pprint layer2)
;    (-> [[[1.0 0.0 0.0 0.0]]]
;        (mlp/padding 2 2 2 2)
;        (mlp/conv-fw (reverse (map reverse layer2)) true)
;        clojure.pprint/pprint)))
