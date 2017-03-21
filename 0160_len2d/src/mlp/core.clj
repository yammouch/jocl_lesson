; lein run  4 4 10001 0.1 1 3 32 # converges

(ns mlp.core
  (:gen-class)
  (:require [mlp.field :as fld]))

(import '(org.jocl CL)
        '(java.util Date))

(require 'mlp.cl)
(alias 'cl 'mlp.cl)

(require 'mlp.mlp-cl)
(alias 'mlp-cl 'mlp.mlp-cl)

(defn one-hot [field-size i]
  (assoc (vec (repeat field-size 0)) (dec i) 1))

(defn a-field [field-size i j]
  (loop [k i acc (vec (repeat field-size 0))]
    (if (<= j k)
      acc
      (recur (inc k) (assoc acc k 1))
      )))

(defn xorshift [x y z w]
  (let [t  (bit-xor x (bit-shift-left x 11))
        wn (bit-and 0xFFFFFFFF
                    (bit-xor w (bit-shift-right w 19)
                             t (bit-shift-right t  8)))]
    (cons w (lazy-seq (xorshift y z w wn)))))

(defn make-input-labels [field-size max-len]
  (let [{ctx :context} @mlp-cl/cl-env
        confs (for [d [:v :h]
                    [start stop] (fld/start-stops field-size max-len)
                    q (range field-size)]
                [start stop q d])]
    [(mapv (comp (partial cl/create-buffer ctx :f)
                 (partial apply concat)
                 (partial apply fld/field1 field-size))
           confs)
     (mapv (comp (partial cl/create-buffer ctx :f)
                 (fn [[start stop]]
                   (fld/one-hot field-size
                                (+ stop (- start) 1))))
           confs)]))

(defn make-minibatches [sb-size in-nd lbl-nd]
  (map (fn [idx] [(mapv in-nd idx) (mapv lbl-nd idx)])
       (partition sb-size (map #(mod % (count in-nd))
                               (xorshift 2 4 6 8)
                               ))))

(defn make-mlp-config [max-len fs cs cd]
  ; fs: field-size, cs: conv size, cd: conv depth
  (let [cs-h (quot cs 2)
        cosize (* cd (+ fs (if (even? cs) 1 0)))] ; conv out size
    [{:type :conv
      :size  [cs cs cd]
      :isize [fs fs  1]
      :pad [cs-h cs-h cs-h cs-h]}
     {:type :sigmoid       :size [(* cosize cosize)]}
     {:type :dense         :size [(* cosize cosize) max-len]}
     {:type :offset        :size [max-len]}
     {:type :softmax       :size [max-len]}
     {:type :cross-entropy :size [max-len]}]))

(defn main-loop [iter learning-rate in-nd lbl-nd]
  (loop [i 0
         [[inputs labels] & bs] (make-minibatches 16 in-nd lbl-nd)
         err-acc (repeat 4 1.0)]
    (if (< iter i)
      :done
      (do
        (mlp-cl/run-minibatch inputs labels learning-rate)
        (if (= (mod i 200) 0)
          (let [err (mlp-cl/fw-err-subbatch in-nd lbl-nd)]
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
        _ (mlp-cl/init
           (make-mlp-config max-len field-size conv-size conv-depth)
           seed)
        [in-nd lbl-nd] (make-input-labels field-size max-len)]
    (main-loop iter learning-rate in-nd lbl-nd)
    (doseq [m [in-nd lbl-nd]] (mlp-cl/release-mem m))
    (mlp-cl/finalize)
    (let [end-time (Date.)]
      (println "end  : " (.toString end-time))
      (printf "%d seconds elapsed\n"
              (quot (- (.getTime end-time) (.getTime start-time))
                    1000))
      (flush))))
