(ns mlp.mid03
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]
            [clojure.pprint]))

(defn mapd [d f s & ss]
  (if (<= d 0)
    (apply f s ss)
    (apply mapv (partial mapd (- d 1) f) s ss)))

(defn swap-dimension [org n t]
  (mapd org (partial apply mapd n vector) t))

(defn read-param [fname]
  (let [[x & xs] (read-string (str "(" (slurp fname) ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] param]
      (alter mlp/jk-mem #(assoc-in % [i :p] (float-array p)))
      )))

(defn print-matrices [ms]
  (doseq [rows (apply map vector ms)]
    (->> rows
         (map (fn [row]
                (concat [""]
                        (map (partial format "%.2f") row))))
         (apply concat)
         (interpose ",")
         (apply str)
         println)))

(defn bw-dense [{bp :b} {b :b p :p [cr cc] :size}]
  (when bp
    (JKernel/mul_mv cr cc bp p b)))

(defn bw-offset [{bp :b} {b :b [n] :size}]
  (when bp
    (System/arraycopy b 0 bp 0 n)))

(defn bw-conv
  [{bp :b}
   {[h w d] :size [ih iw id] :isize [pu _ pl _] :pad
    b :b p :p :as l}]
  (let [ow (mlp/conv-ow l) oh (mlp/conv-oh l)]
    (when bp
      (JKernel/conv_bw_b ih iw oh ow d h w id (- h 1 pu) (- w 1 pl) bp b p)
      )))

(defn bw1
 [{               bp :b                     :as lp} ; previous layer
  {t  :type       b  :b [cr :as size] :size :as l }
  {tn :type in :i bn :b                           } ; next layer
  lr] ; learning-rate
  (case tn
    :cross-entropy
    (case t
      :sigmoid (JKernel/cross_entropy_bw cr             bp in bn lr)
      :softmax (JKernel/cross_entropy_bw (apply + size) bp in bn lr))
    (case t
      :dense   (bw-dense  lp l)
      :offset  (bw-offset lp l)
      :conv    (bw-conv   lp l)
      :sigmoid (JKernel/sigmoid_bw cr             bp in b)
      :softmax (JKernel/sigmoid_bw (apply + size) bp in b)
      )))

(defn bw [b0 label learning-rate]
  (let [layers (->> (-> @mlp/jk-mem
                        (assoc-in [(- (count @mlp/mlp-config) 1) :b] label))
                    (mapv into @mlp/mlp-config))]
    (doseq [[lp l ln] (->> (cons {:b b0} layers)
                           (partition 3 1)
                           (reverse))]
      (bw1 lp l ln learning-rate))))

(def label
  (float-array
   [0 1
    0 0 0 1 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0]))

(defn calc-error []
  (let [out ((last @mlp/jk-mem) :i)]
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out label))))

(defn -main []
  (let [param-fname "data/0_2_4_6_7_8_9_10_11_12_13_15_18_20_22_23_25_26_28_29_30_32_33_35.prm"
        [mlp-config params] (read-param param-fname)
        b0 (float-array 600)
        i0 (float-array 600)]
    (mlp/init mlp-config 0)
    (set-param params)
    (dotimes [_ 50]
      (mlp/fw i0)
      (prn (calc-error))
      (bw b0 label 0.1)
      (JKernel/sub 600 0.9999 i0 i0 b0))
    (->> (reduce #(partition %2 %1) i0 [6 10])
         (swap-dimension 1 1) ; [h w d] -> [h d w]
         (swap-dimension 0 1) ;         -> [d h w]
         (print-matrices))))
