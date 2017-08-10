; - (done) Reduce layers.
; - (done) Replace input data.
; - (done) Add error calculation.
; - (done) Calculate gradient of input vector.
; - Add updating of input vector.
; - Add loop.
; - Refactor matrix dumping.

(ns mlp.mid01
  (:gen-class)
  (:require [mlp.mlp-jk :as mlp]
            [clojure.pprint]))

(defn read-param [fname]
  (let [[x & xs] (read-string (str "(" (slurp fname) ")"))]
    [x (partition 2 xs)]))

(defn set-param [param]
  (dosync
    (doseq [[i p] (take 1 param)]
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

(defn fw []
  (let [nn-input (float-array (* 10 10 6))]
    (mlp/fw nn-input)
    (->> nn-input
         (partition 6)
         (apply map vector)
         (map (partial partition 10))
         print-matrices)
    (doseq [ms (->> (get-in @mlp/jk-mem [0 :p])
                    (partition 24)
                    (apply map vector)
                    (partition 4))]
      (print-matrices (map (partial partition 3) ms)))
    (println (apply str (interpose "," (map (partial format "%.2f")
                                            (:i (last @mlp/jk-mem))
                                            ))))))

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
  (-> (reduce #(vec (repeat %2 %1)) 0 [4 10 10])
      (assoc-in [4 4 0] 1)
      flatten
      float-array))

(defn calc-error []
  (let [out ((last @mlp/jk-mem) :i)]
    (apply + (map #(let [diff (- %1 %2)] (* diff diff))
                  out label))))

(defn -main []
  (let [param-fname "data/0_2_4_6_7_8_9_10_11_12_13_15_18_20_22_23_25_26_28_29_30_32_33_35.prm"
        [_ params] (read-param param-fname)
        b0 (float-array 600)]
    (mlp/init
     [{:type :conv, :size [3 3 4], :isize [10 10 6], :pad [1 1 1 1]}
      {:type :sigmoid, :size [400]}
      {:type :cross-entropy, :size [400]}]
     0)
    (set-param params)
    (fw)
    (prn (calc-error))
    (bw b0 label 0.1)
    (prn (seq b0))))
