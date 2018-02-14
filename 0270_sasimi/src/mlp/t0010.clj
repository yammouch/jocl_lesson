; lein run -m mlp.t0010 1000

(ns mlp.t0010
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.util :as utl]
            [mlp.schemmlp]
            [mlp.meander]
            [mlp.mlp-jk :as mlp]
            [clojure.pprint]))

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

(defn main-loop [iter learning-rate regu tr ts]
  (loop [i 0
         [b & bs] (mlp.schemmlp/make-minibatches 16 tr)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp/run-minibatch (map :field b) (map :cmd b) learning-rate regu)
        (if (= (mod i 100) 0)
          (let [err (map mlp/fw-err (map :field ts) (map :cmd ts))]
            (printf "i: %6d err-avg: %10.6f err-max: %10.6f\n"
                    i
                    (/ (apply + err) (count ts))
                    (apply max err))
            (flush)
            ;(if (every? (partial > 0.02) (cons err err-acc))
            (if false
              :done
              (recur (+ i 1) bs (take 4 (cons err err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn -main [iter]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        iter (read-string iter)
        height 14, width 14
        mlp-config (make-mlp-config 3 4 height width)
        _ (mlp/init mlp-config 1)
        tr (mapv #(mlp.schemmlp/make-input-label % height width)
                 ;(apply concat (mlp.meander/meander-pos 20)))
                 (apply concat (mlp.meander/meander-pos 4)))
        ;[tr ts] (map vec (utl/select tr [16 4] (utl/xorshift 2 4 6 8)))]
        ];[tr ts] (map vec (utl/select tr [4 4] (utl/xorshift 2 4 6 8)))]
    (main-loop iter 0.1 0.9999 tr tr)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
