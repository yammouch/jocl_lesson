(ns mlp.meander-test
  (:require [clojure.test :refer :all]
            [clojure.string]
            [mlp.meander :refer :all]
            [clojure.pprint]
            ))

(defn format-field [field]
  (mapv (fn [row]
          (as-> row r
                (map #(as-> (reverse %) y
                            (reduce (fn [acc x] (+ (* acc 2) x)) y)
                            (if (= y 0) "  " (format "%02X" y)))
                     r)
                (interpose "," r)
                (apply str r)))
        field))

(defn parse-cell [cell]
  (as-> cell c
        (clojure.string/trim c)
        (if (empty? c) 0 (Integer/parseInt c 16))
        (iterate (fn [[_ q]] [(rem q 2) (quot q 2)]) [0 c])
        (mapv first (take 6 (rest c)))))

(defn parse-line [s]
  (mapv (fn [cell] (parse-cell (apply str (rest cell))))
        (re-seq #",[^,]*" (apply str (cons \, s)))
        ))

(deftest meander-0-0-test
  (let [exp (mapv parse-line
                  ;  0              5             10
                  ["0A,02,02,02,01,  ,  ,  ,  ,  ,  ,  ,  ,  " ; 0
                   "  ,  ,  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,03,02,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,02,02,02,20,  ,02,02,02,10,  ,  ,  " ; 5
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  " ; 10
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "])]
    ;(clojure.pprint/pprint
    ; (format-field
    ;  (:field (mlp.meander/meander-0-0 [4 3 2 2 3 3]))))
    (is (= (mlp.meander/meander-0-0 [14 14] [4 3 2 2 3 3])
           {:field exp
            :cmd {:cmd :move-x :org [1 4] :dst 2}}))))

(deftest meander-0-1-test
  (let [exp (mapv parse-line
                  ;  0              5             10
                  ["0A,02,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  " ; 0
                   "  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,01,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,02,02,02,20,  ,02,02,02,10,  ,  ,  " ; 5
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  " ; 10
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "])]
    (is (= (mlp.meander/meander-0-1 [14 14] [4 3 2 2 3 3])
           {:field exp
            :cmd {:cmd :move-y :org [0 0] :dst 5}}))))

(deftest ring-0-0-test
  (let [exp (mapv parse-line
                  ;  0              5             10
                  ["  ,  ,  ,03,02,02,01,  ,  ,  ,  ,  ,  ,  " ; 0
                   "  ,  ,  ,01,  ,  ,01,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,02,02,02,03,02,20,  ,02,02,10,  "
                   "  ,  ,  ,  ,  ,  ,01,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,01,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,01,  ,  ,  ,  ,  ,  ,  " ; 5
                   "0A,02,02,02,02,02,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  " ; 10
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "
                   "  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  ,  "])]
    ;(clojure.pprint/pprint
    ; (format-field 
    ;  (:field (mlp.meander/ring-0-0 [14 14] [6 -4 -2 3 2 2]))))
    (is (= (mlp.meander/ring-0-0 [14 14] [6 -4 -2 3 2 2])
           {:field exp
            :cmd {:cmd :move-y :org [0 4] :dst 2}}))))
