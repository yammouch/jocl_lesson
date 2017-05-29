(ns mlp.schem4
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
  (mapcat #(take 3 (concat (radix %) (repeat 0)))
          (apply concat body)))

(def schem1
 {:field {:body [[0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 3 2 1 0 0 0]
                 [0 0 0 0 1 0 1 0 0 0]
                 [0 0 2 2 0 0 2 2 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]]}
  :cmd {:cmd :move-y :org [5 3] :dst 5}})

(def schem2
 {:field {:body [[0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 1 0 0 0]
                 [0 0 0 0 0 0 1 0 0 0]
                 [0 0 0 0 3 2 0 0 0 0]
                 [0 0 0 0 1 0 0 0 0 0]
                 [0 0 0 0 2 2 1 0 0 0]
                 [0 0 0 0 0 0 1 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]]}
  :cmd {:cmd :move-x :org [4 5] :dst 6}})

(def schem3
 {:field {:body [[0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 2 2 2 1 0 0 0 0]
                 [0 0 0 0 0 1 0 0 0 0]
                 [0 0 0 0 0 7 2 2 0 0]
                 [0 0 0 0 0 1 0 0 0 0]
                 [0 0 0 0 0 3 2 2 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]]}
  :cmd {:cmd :move-y :org [3 2] :dst 4}})

(def schem4
 {:field {:body [[0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]
                 [0 0 0 0 0 3 2 2 0 0]
                 [0 0 0 0 0 1 0 0 0 0]
                 [0 0 0 0 0 7 2 2 0 0]
                 [0 0 0 0 0 1 0 0 0 0]
                 [0 0 2 2 2 0 0 0 0 0]
                 [0 0 0 0 0 0 0 0 0 0]]}
  :cmd {:cmd :move-y :org [3 8] :dst 4}})

(defn make-input-labels [seed]
  (let [rnd (apply mlp/xorshift
             (take 4 (iterate (partial + 2) (+ seed 1))))
        confs1 (smp/expand schem1)
        ts-idx1 (rand-nodup 4 (count confs1) rnd)
        [ts1 tr1] (select confs1 ts-idx1)
        confs2 (smp/expand schem2)
        ts-idx2 (rand-nodup 4 (count confs2) (drop 4 rnd))
        [ts2 tr2] (select confs2 ts-idx2)
        confs3 (smp/expand schem3)
        ts-idx3 (rand-nodup 4 (count confs3) (drop 8 rnd))
        [ts3 tr3] (select confs3 ts-idx3)
        confs4 (smp/expand schem4)
        ts-idx4 (rand-nodup 4 (count confs4) (drop 12 rnd))
        [ts4 tr4] (select confs4 ts-idx4)]
    [(mapv (comp float-array mlp-input-field :field)
           (concat confs1 confs2 confs3 confs4))
     (mapv (comp float-array #(smp/mlp-input-cmd % [10 10]) :cmd)
           (concat confs1 confs2 confs3 confs4))
     (mapv (comp float-array mlp-input-field :field)
           (concat ts1 ts2 ts3 ts4))
     (mapv (comp float-array #(smp/mlp-input-cmd % [10 10]) :cmd)
           (concat ts1 ts2 ts3 ts4))]))

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
      :isize [10 10 3]
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
