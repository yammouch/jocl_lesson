(ns mlp.schem1-tt
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

(def schem
 {:field {:body [[0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0]
                 [0 0 0 3 2 1 0 0]
                 [0 0 0 1 0 1 0 0]
                 [0 2 2 0 0 2 2 0]
                 [0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0]]}
  :cmd {:cmd :move-y :org [4 2] :dst 4}})

(defn make-input-labels []
  (let [confs (smp/expand schem)]
    [(mapv (comp float-array smp/mlp-input-field :field)
           confs)
     (mapv (comp float-array #(smp/mlp-input-cmd % [8 7]) :cmd)
           confs)]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (mlp/xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [cs cd]
  ; cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        co-h (mapv (partial + 7) (if (even? cs) [1 2] [0 0]))
        co-w (mapv (partial + 8) (if (even? cs) [1 2] [0 0]))]
    [{:type :conv
      :size  [cs cs cd]
      :isize [7 8 2]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 0) (co-w 0))]}
     {:type :conv
      :size  [cs cs cd]
      :isize [(co-h 0) (co-w 0) cd]
      :pad   [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 1) (co-w 1))]}
     {:type :dense         :size [(* cd (co-h 1) (co-w 1))
                                  (+ 2 8 7 8)]}
     {:type :offset        :size [(+ 2 8 7 8)]}
     {:type :softmax       :size [   2 8 7 8 ]}
     {:type :cross-entropy :size [(+ 2 8 7 8)]}]))

(defn main-loop [iter learning-rate in-tr lbl-tr in-ts lbl-ts]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 4 in-tr lbl-tr)
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
        [in-nd lbl-nd] (make-input-labels)
        ts-idx (rand-nodup 4 (count in-nd)
                (apply mlp/xorshift
                 (take 4 (iterate (partial + 2) (+ seed 1)))
                 ))
        [in-ts  in-tr ] (select in-nd  ts-idx)
        [lbl-ts lbl-tr] (select lbl-nd ts-idx)]
    ;(dosync (ref-set mlp/debug true))
    (main-loop iter learning-rate in-tr lbl-tr in-ts lbl-ts)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
