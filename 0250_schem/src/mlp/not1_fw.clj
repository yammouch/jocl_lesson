(ns mlp.not1-fw
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]
            [mlp.schemanip :as smp]
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

(defn fw [schem]
  (clojure.pprint/pprint schem)
  (let [parsed (mapv (fn [row] (mapv #(Integer/parseInt % 16)
                                     (re-seq #"\S+" row)))
                     schem)]
    (mlp/fw (float-array (mlp-input-field parsed)))
    (clojure.pprint/pprint (parse-output-vector (:i (last @mlp/jk-mem))))
    (loop [[x & xs] [2 10 10 10], l (:i (last @mlp/jk-mem))]
      (when x
        (println (apply str (interpose " " (map (partial format "%4.2f")
                                                (take x l)))))
        (recur xs (drop x l))
        ))))

(defn format-field [field]
  (clojure.pprint/pprint field)
  (mapv (fn [row]
          (as-> row r
                (map #(->> (reverse %)
                           (reduce (fn [acc x] (+ (* acc 2) x)))
                           (format "%02X"))
                     r)
                (interpose " " r)
                (apply str r)))
        field))

(defn fw2 [schem]
  (mlp/fw (float-array (mapcat (partial apply concat) schem)))
  (let [[cmd from-x from-y to] (parse-output-vector (:i (last @mlp/jk-mem)))]
    (clojure.pprint/pprint [cmd from-x from-y to])
    ((case cmd 0 smp/move-x 1 smp/move-y) schem [from-y from-x] to)))

(defn -main [param-fname schem-fname schem-num]
  (let [schems (read-schem schem-fname (read-string schem-num))
        [mlp-config params] (read-param param-fname)]
    (mlp/init mlp-config 0)
    (set-param params)
    (loop [i 0
           schem (mapv (fn [row]
                         (mapv #(as-> % x
                                 (Integer/parseInt x 16)
                                 (radix x)
                                 (concat x (repeat 0))
                                 (take 6 x)
                                 (vec x))
                               (re-seq #"\S+" row)))
                       (first schems))]
      (when (or (< i 100) schem)
        (clojure.pprint/pprint (format-field schem))
        (recur (+ i 1) (fw2 schem))
        ))))
