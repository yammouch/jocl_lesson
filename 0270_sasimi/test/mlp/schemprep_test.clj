(ns mlp.schemprep-test
  (:require [clojure.test :refer :all]
            [clojure.string]
            [mlp.schemprep]
            [clojure.pprint]
            ))

(defn format-field [field]
  (mapv (fn [row]
          (as-> row r
                (map #(->> (reverse %)
                           (reduce (fn [acc x] (+ (* acc 2) x)))
                           (format "%02X"))
                     r)
                (interpose " " r)
                (apply str r)))
        field))

(defn parse-cell [cell]
  (as-> cell c
        (filter (set "0123456789ABCDEFabcdef") c)
        (if (empty? c) 0 (Integer/parseInt (apply str c) 16))
        (iterate (fn [[_ q]] [(rem q 2) (quot q 2)]) [0 c])
        (mapv first (take 6 (rest c)))))

(defn parse-line [s]
  (mapv (fn [cell] (parse-cell (apply str (rest cell))))
        (re-seq #",[^,]*" (apply str (cons \, s)))
        ))

(deftest room-test
  (let [fld (mapv parse-line
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
    (is (= (mlp.schemprep/room fld)
           [0 8 0 3]))))

(deftest slide-1d-test
  (let [pattern
        [".........." ".........." ".........." ".........." ".........."
         ".........." ".........." ".........." "....221..." ".........."
         ".........." ".........." ".........." "......1..." ".........."
         "....221..." ".221......" "......221." "......1..." ".........."
         "......1..." "...1......" "........1." ".........." "....221..."
         "......1..." "...1......" "........1." ".........." "......1..."
         ".........." ".........." ".........." ".........." "......1..."
         ".........." ".........." ".........." ".........." ".........."]
        [fld ex1 ex2 ex3 ex4]
        (as-> pattern x
              (map (partial mapv #(parse-cell (str %))) x)
              (partition 5 x)
              (apply map vector x))]
    (is (= (mlp.schemprep/slide-1d fld -3 1) ex1))
    (is (= (mlp.schemprep/slide-1d fld  2 1) ex2))
    (is (= (mlp.schemprep/slide-1d fld -2 0) ex3))
    (is (= (mlp.schemprep/slide-1d fld  1 0) ex4))))
