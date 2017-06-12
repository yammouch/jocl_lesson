; lein run -m mlp.not1 10001 0.1 1 3 4 # converges
(ns mlp.not1
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.schemanip :as smp]
            [mlp.mlp-jk :as mlp]))

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

(defn select [coll idx]
  (loop [i 0 [x & xs] coll hit [] pass []]
    (cond (<= (count coll) i) [hit pass]
          (idx i) (recur (+ 1 i) xs (conj hit x) pass)
          :else   (recur (+ 1 i) xs hit (conj pass x))
          )))

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
       (partition 2)
       (map (fn [[field cmd]]
              {:field {:body (mapv (fn [row] (mapv #(Integer/parseInt % 16)
                                                   (re-seq #"\S+" row)))
                                   field)}
               :cmd cmd}))))

(defn make-input-labels [seed]
  (let [rnd (apply mlp/xorshift
             (take 4 (iterate (partial + 2) (+ seed 1))))
        confs (map (fn [rnd x]
                     (let [expanded (smp/expand x)]
                       [expanded
                        (select expanded
                                (rand-nodup (count rnd) (count expanded) rnd)
                                )]))
                   (partition 4 rnd)
                   schems)]
    [(mapv (comp float-array mlp-input-field :field)
           (mapcat first confs))
     (mapv (comp float-array #(smp/mlp-input-cmd % [10 10]) :cmd)
           (mapcat first confs))
     (mapv (comp float-array mlp-input-field :field)
           (mapcat (comp first second) confs))
     (mapv (comp float-array #(smp/mlp-input-cmd % [10 10]) :cmd)
           (mapcat (comp first second) confs))]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (mlp/xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [cs cd]
  ; cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        co-h (mapv (partial + 10) (if (even? cs) [1 2] [0 0]))
        co-w (mapv (partial + 10) (if (even? cs) [1 2] [0 0]))]
    [{:type :conv
      :size  [cs cs cd]
      :isize [10 10 5]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 0) (co-w 0))]}
     {:type :conv
      :size  [cs cs cd]
      :isize [(co-h 0) (co-w 0) cd]
      :pad   [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 1) (co-w 1))]}
     {:type :dense         :size [(* cd (co-h 1) (co-w 1))
                                  (+ 2 10 10 10)]}
     {:type :offset        :size [(+ 2 10 10 10)]}
     {:type :softmax       :size [   2 10 10 10 ]}
     {:type :cross-entropy :size [(+ 2 10 10 10)]}]))

(defn main-loop [iter learning-rate in-tr lbl-tr in-ts lbl-ts]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 16 in-tr lbl-tr)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp/run-minibatch inputs labels learning-rate)
        (if (= (mod i 500) 0)
          (let [err (mlp/fw-err-subbatch in-ts lbl-ts)]
            (printf "i: %6d err: %8.2f\n" i err) (flush)
            (if (every? (partial > 0.02) (cons err err-acc))
              :done
              (recur (+ i 1) bs (take 4 (cons err err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn -main [& args]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        [iter learning-rate seed conv-size conv-depth]
        (mapv read-string args)
        _ (mlp/init
           (make-mlp-config conv-size conv-depth)
           seed)
        [in-tr lbl-tr in-ts lbl-ts] (make-input-labels seed)]
    ;(dosync (ref-set mlp/debug true))
    (main-loop iter learning-rate in-tr lbl-tr in-ts lbl-ts)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
