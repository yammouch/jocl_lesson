; lein run -m mlp.t0150-train 100000 data/t0150_train.dat

(ns mlp.t0150-train
  (:gen-class)
  (:import  [java.util Date])
  (:require [clojure.pprint]
            [mlp.schemmlp]
            [mlp.mlp-jk :as mlp]))

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
     ;{:type :softmax       :size [   2 w h (max w h) ]}
     {:type :sigmoid       :size [(+ 2 w h (max w h))]}
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
          (let [err (map mlp/fw-err (map :field ts) (map :cmd ts))
                err-reduced (apply max err)
                ];err-tr (map mlp/fw-err (map :field tr) (map :cmd tr))]
            (printf "i:%6d" i)
            ;(printf " err-tr avg:%6.3f max:%6.3f"
            ;        (/ (apply + err-tr) (count tr))
            ;        (apply max err-tr))
            (printf " err-ts avg:%6.3f max:%6.3f"
                    (/ (apply + err) (count ts))
                    (apply max err))
            (printf "\n")
            (flush)
            (if (every? (partial > 0.02) (cons err-reduced err-acc))
              :done
              (recur (+ i 1) bs (take 4 (cons err-reduced err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn parse-float-array [pairs]
  (mapv #(as-> % x
               (update-in x [:field] float-array)
               (update-in x [:cmd  ] float-array))
        pairs))

(defn -main [iter tr-file-name]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        iter (read-string iter)
        height 10, width 10
        mlp-config (make-mlp-config 3 3 height width)
        _ (mlp/init mlp-config 2)
        [tr ts] (read-string (str "(" (slurp tr-file-name) ")"))
        [tr ts] (map parse-float-array [tr ts])]
    (main-loop iter 0.1 0.9999 tr ts)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
