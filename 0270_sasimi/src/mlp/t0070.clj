; lein run -m mlp.t0070 40000

(ns mlp.t0070
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.util :as utl]
            [mlp.schemmlp]
            [mlp.schemprep :as spp]
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
          (let [err (map mlp/fw-err (map :field ts) (map :cmd ts))
                err-reduced (apply max err)
                err-tr (map mlp/fw-err (map :field tr) (map :cmd tr))]
            (printf "i:%6d" i)
            (printf " err-tr avg:%6.3f max:%6.3f"
                    (/ (apply + err-tr) (count tr))
                    (apply max err-tr))
            (printf " err-ts avg:%6.3f max:%6.3f"
                    (/ (apply + err) (count ts))
                    (apply max err))
            (printf "\n")
            (flush)
            (if (every? (partial > 0.02) (cons err-reduced err-acc))
              :done
              (recur (+ i 1) bs (take 4 (cons err-reduced err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn meander-0-pos [size g]
  (let [m (vec (mlp.meander/meander-0 size g))
        [u d l r] (spp/room (get-in m [0 :field]))
        ml (for [dy (range (- u) (+ d 1)) dx (range (- l) (+ r 1))]
             [dy dx])
        n (count ml)
        [mtr mts] (utl/select (vec ml) [(- n 1) 1] (utl/xorshift 2 4 6 8))]
    [(mapv (partial mlp.schemmlp/slide-history m) mtr)
     (mapv (partial mlp.schemmlp/slide-history m) mts)]))

(defn ring-0-pos [size g]
  (let [m (vec (mlp.meander/ring-0 size g))
        [u d l r] (spp/room (get-in m [0 :field]))
        ml (for [dy (range (- u) (+ d 1)) dx (range (- l) (+ r 1))]
             [dy dx])
        n (count ml)
        [mtr mts] (utl/select (vec ml) [(- n 1) 1] (utl/xorshift 3 6 9 12))]
    [(mapv (partial mlp.schemmlp/slide-history m) mtr)
     (mapv (partial mlp.schemmlp/slide-history m) mts)]))

(defn test-pattern [size]
  (as-> (concat (for [g0 [2    ] g1 [2 3 4] g2 [2 3 4]
                      g3 [2    ] g4 [2    ] g5 [2    ]]
                  (meander-0-pos size [(+ g0 g2) g1 g2 g3 g4 g5]))
                (for [g0 [ 2    ] g1 [-4 -3 -2] g2 [2 3 4]
                      g3 [-2    ] g4 [2       ] g5 [2    ]]
                  (meander-0-pos size [(+ g0 g2) g1 g2 g3 g4 g5]))
                (for [g0 [4    ] g1 [4    ] g2 [2 3 4]
                      g3 [2 3 4] g4 [2    ] g5 [2    ]]
                  (ring-0-pos size [g0 g1 g2 g3 g4 g5]))) s
        (apply map vector s)))

(defn -main [iter]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        iter (read-string iter)
        height 14, width 14
        mlp-config (make-mlp-config 3 4 height width)
        _ (mlp/init mlp-config 2)
        p (test-pattern [height width])
        [tr ts] (apply map (comp (partial apply concat)
                                 (partial apply concat)
                                 vector)
                           p)
        tr (mapv #(mlp.schemmlp/make-input-label % height width) tr)
        ts (mapv #(mlp.schemmlp/make-input-label % height width) ts)]
    (main-loop iter 0.1 0.9999 tr ts)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
