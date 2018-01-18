(ns mlp.meander
  (:gen-class)
  (:require [mlp.parse-csv :as psc]))

(defn is-delimiter? [row]
  (= (take 6 (first row)) (seq "#start")))

(defn -main [& args]
  (as-> (slurp "data/meander.csv") x
        (remove (partial = \return) x)
        (first (psc/csv x))
        (iterate #(as-> (drop-while           is-delimiter?  %) y
                        (drop-while (comp not is-delimiter?) y))
                 x)
        (take-while (comp not empty?) x)
        (map #(split-with (fn [row] (not= (ffirst row) \:)) %) x)
        (map (fn [[field cmd]]
               [(as-> (rest field) fld
                      (map (fn [row] (vec (rest row))) fld)
                      (vec fld))
                (ffirst cmd)])
             x)
        (clojure.pprint/pprint x)))
