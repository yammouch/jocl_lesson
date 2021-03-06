(ns mlp.not1-fw-mid
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]
            [clojure.pprint]))

(defn radix [x]
  (loop [x x acc []]
    (if (<= x 0)
      acc
      (recur (quot x 2) (conj acc (rem x 2)))
      )))

(defn mlp-input-field [body]
  (mapcat #(take 6 (concat (radix %) (repeat 0)))
          (apply concat body)))

(defn read-schem [fname num]
  (->> (read-string (str "(" (slurp fname) ")"))
       (partition 3)
       (filter (comp #{num} first))
       (map second)))

(defn read-param [fname]
  (let [[x & xs] (read-string (str "(" (slurp fname) ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] param]
      (alter mlp/jk-mem #(assoc-in % [i :p] (float-array p)))
      )))

(defn print-matrices [ms]
  (doseq [rows (apply map vector ms)]
    (->> rows
         (map (fn [row]
                (concat [""]
                        (map (partial format "%.2f") row))))
         (apply concat)
         (interpose ",")
         (apply str)
         println)))

(defn fw [schem]
  (clojure.pprint/pprint schem)
  (let [parsed (mapv (fn [row] (mapv #(Integer/parseInt % 16)
                                     (re-seq #"\S+" row)))
                     schem)
        nn-input (float-array (mlp-input-field parsed))]
    (mlp/fw nn-input)
    (->> nn-input
         (partition 6)
         (apply map vector)
         (map (partial partition 10))
         print-matrices)
    (doseq [ms (->> (get-in @mlp/jk-mem [0 :p])
                    (partition 24)
                    (apply map vector)
                    (partition 4))]
      (print-matrices (map (partial partition 3) ms)))
    (->> (get-in @mlp/jk-mem [2 :i])
         (partition 4)
         (apply map vector)
         (map (partial partition 10))
         print-matrices)
    (doseq [ms (->> (get-in @mlp/jk-mem [2 :p])
                    (partition 16)
                    (apply map vector)
                    (partition 4))]
      (print-matrices (map (partial partition 3) ms)))
    (->> (get-in @mlp/jk-mem [4 :i])
         (partition 4)
         (apply map vector)
         (map (partial partition 10))
         print-matrices)
    (doseq [ms (->> (get-in @mlp/jk-mem [4 :p])
                    (partition 32)
                    (apply map vector)
                    (map (fn [v] (->> (partition 4 v)
                                      (apply map vector)
                                      (map (partial partition 10)))))
                    (apply map vector))]
      (print-matrices ms))
    (->> (get-in @mlp/jk-mem [5 :p])
         (map (partial format "%.2f"))
         (interpose ",")
         (apply str)
         println)
    (loop [[x & xs] [2 10 10 10], l (:i (last @mlp/jk-mem))]
      (when x
        (println (apply str (interpose "," (map (partial format "%.2f")
                                                (take x l)))))
        (recur xs (drop x l))
        ))))

(defn -main []
  (let [param-fname "data/0_2_4_6_7_8_9_10_11_12_13_15_18_20_22_23_25_26_28_29_30_32_33_35.prm"
        schem-fname "data/not1.dat"
        schem-num "13"
        schems (read-schem schem-fname (read-string schem-num))
        [mlp-config params] (read-param param-fname)]
    (mlp/init mlp-config 0)
    (set-param params)
    (doseq [s schems] (fw s))
    ))
