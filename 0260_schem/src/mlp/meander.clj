(ns mlp.meander
  (:gen-class)
  (:require [mlp.parse-csv :as psc]))

(defn mapd [f d s & ss]
  (if (<= d 0)
    (apply f s ss)
    (apply mapv (partial mapd f (- d 1)) s ss)))

(defn is-delimiter? [row]
  (= (take 6 (first row)) (seq "#start")))

(defn read-file []
  (as-> (slurp "data/meander.csv") x
        (remove (partial = \return) x)
        (first (psc/csv x))
        (iterate #(drop-while (comp not is-delimiter?) (rest %)) x)
        (take-while (comp not empty?) x)
        (map #(split-with (fn [row] (not= (ffirst row) \:)) %) x)
        (map (fn [[field cmd]]
               [(as-> (rest field) fld
                      (map (fn [row] (vec (rest row))) fld)
                      (vec fld))
                (ffirst cmd)])
             x)
        (map (fn [[field cmd]]
               {:field (mapd (fn [s] (if (empty? s) 0 (Integer/parseInt s 16)))
                             2 field)
                :cmd (let [[cmd [y x] to] (read-string (str "[" cmd "]"))]
                       {:cmd cmd :org [x y] :dst to})})
             x)))

(defn -main [& args]
  (clojure.pprint/pprint (read-file)))
