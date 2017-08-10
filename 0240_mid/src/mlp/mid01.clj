; - (done) Reduce layers.
; - (done) Replace input data.
; - (done) Add error calculation.
; - Calculate gradient of input vector.
; - Add updating of input vector.
; - Add loop.
; - Refactor matrix dumping.

(ns mlp.mid01
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]
            [clojure.pprint]))

(defn read-param [fname]
  (let [[x & xs] (read-string (str "(" (slurp fname) ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] (take 1 param)]
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

(defn fw []
  (let [nn-input (float-array (* 10 10 6))]
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
    (println (apply str (interpose "," (map (partial format "%.2f")
                                            (:i (last @mlp/jk-mem))
                                            ))))))

(defn calc-error []
  (let [out ((last @mlp/jk-mem) :i)]
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out
                 (-> (reduce #(vec (repeat %2 %1)) 0 [4 10 10])
                     (assoc-in [4 4 0] 1)
                     flatten)))))

(defn -main []
  (let [param-fname "data/0_2_4_6_7_8_9_10_11_12_13_15_18_20_22_23_25_26_28_29_30_32_33_35.prm"
        [_ params] (read-param param-fname)]
    (mlp/init
     [{:type :conv, :size [3 3 4], :isize [10 10 6], :pad [1 1 1 1]}
      {:type :sigmoid, :size [400]}
      {:type :cross-entropy, :size [400]}]
     0)
    (set-param params)
    (fw)
    (prn (calc-error))))
