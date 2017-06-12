(ns mlp.not1-fw
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]))

(defn radix [x]
  (loop [x x acc []]
    (if (<= x 0)
      acc
      (recur (quot x 2) (conj acc (rem x 2)))
      )))

(defn mlp-input-field [{body :body}]
  (mapcat #(take 5 (concat (radix %) (repeat 0)))
          (apply concat body)))

(def schems
  (->> (read-string (str "(" (slurp "src/mlp/not1.dat") ")"))
       (partition 3)
       (map (fn [[_ field cmd]]
              {:field {:body (mapv (fn [row] (mapv #(Integer/parseInt % 16)
                                                   (re-seq #"\S+" row)))
                                   field)}
               :cmd cmd}))))

(defn read-param []
  (let [[x & xs] (read-string (str "(" (slurp "param.dat") ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] param]
      (alter mlp/jk-mem #(assoc-in % [i :p] (float-array p)))
      )))

(defn -main [& args]
  (let [[mlp-config params] (read-param)]
    (mlp/init mlp-config 0)
    (set-param params)
    (mlp/fw (float-array (mlp-input-field (:field (nth schems 4)))))
    (println (apply str (map (partial format " %3.1f")
                             (:i (last @mlp/jk-mem))
                             )))))
