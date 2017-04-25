(ns mlp.schemanip-test
  (:require [clojure.test  :refer :all]
            [mlp.schemanip :as smp]
            [clojure.pprint]))

(deftest test-slide-upper
  (let [test-patterns
        [{:arg  {:field {:size [2 3]
                         :body [[0 0]
                                [1 0]
                                [0 0]]}
                 :cmd {:cmd :move-y :org [1 2] :dst 1}}
          :expc {:field {:size [2 3]
                         :body [[1 0]
                                [0 0]
                                [0 0]]}
                 :cmd {:cmd :move-y :org [1 1] :dst 0}}}
         {:arg  {:field {:size [2 3]
                         :body [[0 0]
                                [1 0]
                                [0 0]]}
                 :cmd {:cmd :move-y :org [1 2] :dst 0}}
          :expc nil
          }]]
    (doseq [{arg :arg expc :expc} test-patterns]
      (let [result (smp/slide-upper arg)]
        (is (= result expc))))))

(deftest test-slide-lower
  (let [test-patterns
        [{:arg  {:field {:size [2 3]
                         :body [[0 0]
                                [1 0]
                                [0 0]]}
                 :cmd {:cmd :move-y :org [1 1] :dst 0}}
          :expc {:field {:size [2 3]
                         :body [[0 0]
                                [0 0]
                                [1 0]]}
                       :cmd {:cmd :move-y :org [1 2] :dst 1}}}
         {:arg  {:field {:size [2 3]
                         :body [[0 0]
                                [0 0]
                                [0 1]]}
                 :cmd {:cmd :move-y :org [1 1] :dst 0}}
          :expc nil}]]
    (doseq [{arg :arg expc :expc} test-patterns]
      (let [result (smp/slide-lower arg)]
        (is (= result expc))))))

;(deftest test-slide-left
;  (let [test-patterns
;        [{:arg  {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 0] [0 1]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-x :org [1 1] :dst 2}}
;          :expc {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 1] [0 0]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-x :org [0 1] :dst 1}}}
;         {:arg  {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 1] [0 0]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-y :org [1 1] :dst 2}}
;          :expc nil}]]
;    (doseq [{arg :arg expc :expc} test-patterns]
;      (let [result (smp/slide-left arg)]
;        (is (= result expc))))))
;
;(deftest test-slide-right
;  (let [test-patterns
;        [{:arg  {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 0] [0 1]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-x :org [1 1] :dst 2}}
;          :expc {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 1] [0 0]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-x :org [0 1] :dst 1}}}
;         {:arg  {:field {:size [2 3]
;                         :body [[[0 0] [0 0]]
;                                [[0 1] [0 0]]
;                                [[0 0] [0 0]]]}
;                 :cmd {:cmd :move-y :org [1 1] :dst 2}}
;          :expc nil}]]
;    (doseq [{arg :arg expc :expc} test-patterns]
;      (let [result (smp/slide-left arg)]
;        (is (= result expc))))))
;
;(def schem
;  {:field (read-string (slurp "td000.txt"))
;   :cmd   {:cmd :move-y :org [9 6] :dst 9}})
;
;(deftest test-slide
;  (let [test-patterns
;        [{:f smp/slide-upper :expc   7}
;         {:f smp/slide-lower :expc  11}
;         {:f smp/slide-left  :expc   6}
;         {:f smp/slide-right :expc   6}]]
;    (doseq [{f :f expc :expc} test-patterns]
;      (let [result (->> (iterate f schem)
;                        (take-while identity)
;                        (take 1000)
;                        count)]
;        (is (= result expc))))))
;
;(deftest test-expand
;  (let [test-patterns
;        [{:f smp/expand-v    :expc  17}
;         {:f smp/expand-h    :expc  11}
;         {:f smp/expand      :expc 187}]]
;    (doseq [{f :f expc :expc} test-patterns]
;      (let [result (->> (f schem)
;                        (take-while identity)
;                        (take 1000)
;                        count)]
;        (is (= result expc))))))
; 
;(deftest test-mlp-input-field
;  (let [test-patterns
;        [{:arg  {:body [[[1 0] [0 1] [0 1]] [[0 0] [1 0] [0 0]]] :size :na}
;          :expc          [1 0   0 1   0 1     0 0   1 0   0 0]}]]
;    (doseq [{arg :arg expc :expc} test-patterns]
;      (let [result (smp/mlp-input-field arg)]
;        (is (= result expc))))))
;
;(deftest test-mlp-input-cmd
;  (let [test-patterns
;        [{:arg  [{:cmd :move-x :org [1         2] :dst 3}     [4 5]]
;          :expc [ 1 0          0 1 0 0  0 0 1 0 0  0 0 0 1 0]}]]
;    (doseq [{arg :arg expc :expc} test-patterns]
;      (let [result (apply smp/mlp-input-cmd arg)]
;        (is (= result expc))))))
;
;(deftest test-mlp-input
;  (let [test-patterns
;        [{:arg  {:field {:body [[[1 0] [0 1] [0 1]] [[0 0] [1 0] [0 0]]]
;                         :size [3 2]}
;                 :cmd   {:cmd :move-y :org [2 1] :dst 0}}
;          :expc {:niv [1 0  0 1  0 1  0 0  1 0  0 0]
;                 :eov [0 1  0 0 1  0 1  1 0 0]}}]]
;    (doseq [{arg :arg expc :expc} test-patterns]
;      (let [result (smp/mlp-input arg)]
;        (is (= result expc))))))
