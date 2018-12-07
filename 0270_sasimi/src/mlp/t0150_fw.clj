; lein run -m mlp.t0150-fw data/t0150_pr.dat

(ns mlp.t0150-fw
  (:gen-class)
  (:require [clojure.pprint]
            [mlp.mlp-jk :as mlp]
            [mlp.meander]
            [mlp.schemedit :as smp]))

(defn make-schem []
  (:field (first (mlp.meander/ring-0 [10 10] [4 4 3 3 1 2]))))

(defn read-param [fname]
  (let [[x & xs] (read-string (str "(" (slurp fname) ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] param]
      (alter mlp/jk-mem #(assoc-in % [i :p] (float-array p)))
      )))

(defn split-output-vector [ns l]
  (loop [[x & xs] ns, l l, acc []]
    (if x
      (recur xs (drop x l) (conj acc (take x l)))
      acc)))

(defn decode-one-hot [l]
  (->> (map-indexed vector l)
       (apply max-key #(% 1))
       first))

(defn parse-output-vector [l]
  (mapv decode-one-hot (split-output-vector [2 10 10 10] l)))

(defn format-field [field]
  (mapv (fn [row]
          (as-> row r
                (map #(->> (reverse %)
                           (reduce (fn [acc x] (+ (* acc 2) x)))
                           (format "%02X"))
                     r)
                (interpose " " r)
                (apply str r)))
        field))

(defn fw [schem]
  (mlp/fw (float-array (mapcat (partial apply concat) schem)))
  (let [[cmd from-x from-y to] (parse-output-vector (:i (last @mlp/jk-mem)))]
    (clojure.pprint/pprint [cmd from-x from-y to])
    ((case cmd 0 smp/move-x 1 smp/move-y) schem [from-y from-x] to)))

(defn edit1 [schem]
  (clojure.pprint/pprint schem)
  (loop [i 0
         schem schem
         schem-next (fw schem)]
    (if (and (< i 100) schem-next)
      (recur (+ i 1) schem-next (fw schem-next))
      (clojure.pprint/pprint (format-field schem))
      )))

(defn main-loop [schems]
  (doseq [s schems]
    (edit1 s)))

(defn -main [param-fname]
  (let [schems [(make-schem)]
        [mlp-config params] (read-param param-fname)]
    (mlp/init mlp-config 0)
    (set-param params)
    (main-loop schems)))
