; lein run -m mlp.t0010 1000
(ns mlp.t0010
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.schemprep :as smp]
            [mlp.meander]
            [mlp.mlp-jk :as mlp]
            [clojure.pprint]
            [clojure.java.io]))

(defn padding [rows h w]
  (let [empty 0]
    (as-> (concat rows (repeat [])) rows 
          (map (fn [row]
                 (as-> (repeat empty) x
                       (concat row x)
                       (take w x)))
               rows)
          (take h rows))))

(defn one-hot [val len]
  (take len (concat (repeat val 0) [1] (repeat 0))))

(defn mlp-input-cmd [{cmd :cmd [x y] :org dst :dst} [cx cy]]
  (concat (case cmd :move-x [1 0] [0 1])
          (one-hot x cx)
          (one-hot y cy)
          (one-hot dst (max cx cy))))

(defn make-input-labels [schems h w seed]
  (let [schems (map (fn [schem] (update-in schem [:field] padding h w))
                    schems)
        confs schems;(map smp/expand schems)
        test-data schems];(select confs seed)]
    [(mapv (comp float-array
                 (partial apply concat)
                 (partial apply concat)
                 :field)
           confs)
     (mapv (comp float-array #(mlp-input-cmd % [w h]) :cmd)
           confs)
     (mapv (comp float-array
                 (partial apply concat)
                 (partial apply concat)
                 :field)
           test-data)
     (mapv (comp float-array #(mlp-input-cmd % [w h]) :cmd)
           test-data)]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (mlp/xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [cs cd h w]
  ; cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        co-h (mapv (partial + h) (if (even? cs) [1 2] [0 0]))
        co-w (mapv (partial + w) (if (even? cs) [1 2] [0 0]))]
    [{:type :conv
      :size  [cs cs cd]
      :isize [h w 6]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 0) (co-w 0))]}
     {:type :conv
      :size  [cs cs cd]
      :isize [(co-h 0) (co-w 0) cd]
      :pad   [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (co-h 1) (co-w 1))]}
     {:type :dense         :size [(* cd (co-h 1) (co-w 1))
                                  (+ 2 w h (max w h))]}
     {:type :offset        :size [(+ 2 w h (max w h))]}
     {:type :softmax       :size [   2 w h (max w h) ]}
     {:type :cross-entropy :size [(+ 2 w h (max w h))]}]))

(defn main-loop [iter learning-rate regu in-tr lbl-tr in-ts lbl-ts]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 16 in-tr lbl-tr)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp/run-minibatch inputs labels learning-rate regu)
        (if (= (mod i 100) 0)
          (let [err (map mlp/fw-err in-ts lbl-ts)]
            (printf "i: %6d err-avg: %10.6f err-max: %10.6f\n"
                    i
                    (/ (apply + err) (count in-ts))
                    (apply max err))
            (flush)
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

(defn -main [iter]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        iter (read-string iter)
        height 14, width 14
        mlp-config (make-mlp-config 3 4 height width)
        _ (mlp/init mlp-config 1)
        [in-tr lbl-tr in-ts lbl-ts]
        (make-input-labels (apply concat (mlp.meander/meander-pos 4))
                           height width 1)]
    (main-loop iter 0.1 0.9999 in-tr lbl-tr in-ts lbl-ts)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
