; lein run -m mlp.not1-large 1000 data/not1.dat data/hoge.dat 0
(ns mlp.not1-large
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.schemprep :as smp]
            [mlp.mlp-jk :as mlp]
            [clojure.pprint]
            [clojure.java.io]))

(defn lift [[x & xs] n]
  (cond (not x) n
        (< n x) n
        :else (recur xs (+ 1 n))
        ))

(defn rand-nodup [n lt rs]
  (loop [acc (sorted-set)
         [x & xs] (map rem rs (range lt (- lt n) -1))]
    (if x
      (recur (conj acc (lift (seq acc) x)) xs)
      acc)))

(defn select [expanded seed]
  (let [rnd (apply mlp/xorshift
             (take 4 (iterate (partial + 2) (+ seed 1))))]
    (mapv (fn [rnd xs]
            (as-> (rand-nodup (count rnd) (count xs) rnd) x
                  (filter (comp x first) (map-indexed vector xs))
                  (mapv second x)))
          (partition 1 rnd)
          expanded)))

(defn radix [x]
  (loop [x x acc []]
    (if (<= x 0)
      acc
      (recur (quot x 2) (conj acc (rem x 2)))
      )))

(defn mlp-input-field [body]
  (mapcat #(take 6 (concat (radix %) (repeat 0)))
          (apply concat body)))

(defn print-training-data [td]
  (with-open [o (clojure.java.io/writer "training_data.dat")]
    (doseq [s td]
      (clojure.pprint/pprint s o))))

(defn read-schems [fname exclude]
  (->> (read-string (str "(" (slurp fname) ")"))
       (partition 3)
       (filter (comp not
                     (->> exclude
                          (re-seq #"\d+")
                          (map read-string)
                          set)
                     first))
       (#(do (print-training-data %) %))
       (map (fn [[_ field cmd]]
              {:field (mapv (fn [row] (mapv #(Integer/parseInt % 16)
                                           (re-seq #"\S+" row)))
                            field)
               :cmd cmd}))))

(defn padding [rows]
  (let [empty 0]
    (as-> (concat rows (repeat [])) rows 
          (map (fn [row]
                 (as-> (repeat empty) x
                       (concat row x)
                       (take 20 x)))
               rows)
          (take 20 rows))))

(defn make-input-labels [schems seed]
  (let [schems (map (fn [schem] (update-in schem [:field] padding))
                    schems)
        confs (map smp/expand schems)
        test-data (select confs seed)]
    [(mapv (comp float-array mlp-input-field :field)
           (apply concat confs))
     (mapv (comp float-array #(smp/mlp-input-cmd % [20 20]) :cmd)
           (apply concat confs))
     (mapv (comp float-array mlp-input-field :field)
           (apply concat test-data))
     (mapv (comp float-array #(smp/mlp-input-cmd % [20 20]) :cmd)
           (apply concat test-data))]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (mlp/xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [cs cd]
  ; cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        co-h (mapv (partial + 20) (if (even? cs) [1 2] [0 0]))
        co-w (mapv (partial + 20) (if (even? cs) [1 2] [0 0]))]
    [{:type :conv
      :size  [cs cs cd]
      :isize [20 20 6]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 0) (co-w 0))]}
     {:type :conv
      :size  [cs cs cd]
      :isize [(co-h 0) (co-w 0) cd]
      :pad   [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 1) (co-w 1))]}
     {:type :dense         :size [(* cd (co-h 1) (co-w 1))
                                  (+ 2 20 20 20)]}
     {:type :offset        :size [(+ 2 20 20 20)]}
     {:type :softmax       :size [   2 20 20 20 ]}
     {:type :cross-entropy :size [(+ 2 20 20 20)]}]))

(defn main-loop [iter learning-rate regu in-tr lbl-tr in-ts lbl-ts]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 16 in-tr lbl-tr)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp/run-minibatch inputs labels learning-rate regu)
        (if (= (mod i 100) 0)
          (let [err (mlp/fw-err-minibatch in-ts lbl-ts)]
            (printf "i: %6d err: %10.6f\n" i (/ err (count in-ts))) (flush)
            ;(if (every? (partial > 0.02) (cons err err-acc))
            (if false
              :done
              (recur (+ i 1) bs (take 4 (cons err err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn print-param [fname cfg m]
  (with-open [o (clojure.java.io/writer fname)]
    (clojure.pprint/pprint cfg o)
    (loop [i 0 [x & xs] m]
      (if x
        (do (when (:p x)
              (clojure.pprint/pprint i o)
              (clojure.pprint/pprint (seq (:p x)) o))
            (recur (+ i 1) xs))
        :done))))

(defn -main [iter schem param exclude]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        iter (read-string iter)
        mlp-config (make-mlp-config 3 4)
        _ (mlp/init mlp-config 1)
        [in-tr lbl-tr in-ts lbl-ts]
        (make-input-labels (read-schems schem exclude) 1)]
    ;(dosync (ref-set mlp/debug true))
    (main-loop iter 0.1 0.9999 in-tr lbl-tr in-ts lbl-ts)
    (print-param param mlp-config @mlp/jk-mem)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
