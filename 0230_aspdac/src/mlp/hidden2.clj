(ns mlp.hidden2
  (:gen-class)
  (:require [mlp.mlp-clj :as mlp]
            [clojure.pprint]))

(defn rmap
 ([f tr] (rmap f tr (comp not seq?)))
 ([f tr atom?]
  (cond (and (seq? tr) (empty? tr)) '()
        (atom? tr) (f tr)
        :else (cons (rmap f (first tr) atom?)
                    (rmap f (rest  tr) atom?)
                    ))))

(defn read-file [fname]
  (let [read-data (->> (slurp fname)
                       (#(str "[" % "]"))
                       read-string)]
    [(reduce #(partition %2 %1) (nth read-data 2) [4 5 3])
     (reduce #(partition %2 %1) (nth read-data 4) [4 4 3])]))

(defn show-l01 [l]
  (let [d0 (rmap #(nth % 0) l #(and (seq? %) (not (seq? (first %)))))]
    (doseq [r d0]
      (->> r
           (map (partial format "%4.1f"))
           (apply println)
           ))))

(defn -main [file-name]
  (let [[l1 l2] (read-file file-name)
        l12 (-> [[[1.0 0.0 0.0 0.0]]]
                (mlp/padding 2 2 2 2)
                (mlp/conv-fw (reverse (map reverse l2)) true))
        l01 (-> l12
                (mlp/padding 2 2 2 2)
                (mlp/conv-fw (reverse (map reverse l1)) true))]
    (clojure.pprint/pprint l2)
    (clojure.pprint/pprint l12)
    (clojure.pprint/pprint l01)
    (show-l01 l01)))
