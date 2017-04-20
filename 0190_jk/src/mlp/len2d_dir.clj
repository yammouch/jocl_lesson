(ns mlp.len2d-dir
  (:gen-class)
  (:import  [java.util Date])
  (:require [mlp.field  :as fld]
            [mlp.mlp-jk :as mlp]))

(defn make-input-labels [field-size max-len]
  (let [confs (for [d [:v :h]
                    [start stop] (fld/start-stops field-size max-len)
                    q (range field-size)]
                [start stop q d])]
    [(mapv (comp float-array
                 (partial apply concat)
                 (partial apply fld/field1 field-size))
           confs)
     (mapv (comp float-array
                 (fn [[start stop _ d]]
                   (concat (case d :v [1 0] :h [0 1])
                           (fld/one-hot field-size
                                        (+ stop (- start) 1)))))
           confs)]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (mlp/xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [max-len fs cs cd]
  ; fs: field-size, cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        cosize (mapv (partial + fs) (if (even? cs) [1 2] [0 0]))]
    [{:type :conv
      :size  [cs cs cd]
      :isize [fs fs  1]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (cosize 0) (cosize 0))]}
     {:type :conv
      :size  [cs cs cd]
      :isize [(cosize 0) (cosize 0) cd]
      :pad   [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cd (cosize 1) (cosize 1))]}
     {:type :dense         :size [(* cd (cosize 1) (cosize 1))
                                  (+ 2 max-len)]}
     {:type :offset        :size [(+ 2 max-len)]}
     {:type :softmax       :size [   2 max-len ]}
     {:type :cross-entropy :size [(+ 2 max-len)]}]))

(defn main-loop [iter learning-rate in-nd lbl-nd]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 4 in-nd lbl-nd)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp/run-minibatch inputs labels learning-rate)
        (if (= (mod i 500) 0)
          (let [err (mlp/fw-err-subbatch in-nd lbl-nd)]
            (printf "i: %6d err: %8.2f\n" i err) (flush)
            (if (every? (partial > 0.02) (cons err err-acc))
              :done
              (recur (+ i 1) bs (take 4 (cons err err-acc)))))
          (recur (+ i 1) bs err-acc))))))

(defn -main [& args]
  (let [start-time (Date.)
        _ (println "start: " (.toString start-time))
        [field-size max-len iter learning-rate seed conv-size conv-depth]
        (mapv read-string args)
        _ (mlp/init
           (make-mlp-config max-len field-size conv-size conv-depth)
           seed)
        [in-nd lbl-nd] (make-input-labels field-size max-len)]
    (dosync (ref-set mlp/debug true))
    (main-loop iter learning-rate in-nd lbl-nd)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
