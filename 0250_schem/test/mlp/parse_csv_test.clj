(ns mlp.parse-csv-test
  (:require [clojure.test :refer :all]
            [mlp.parse-csv :as csv]
            [clojure.pprint]))

(deftest test-cell
  (is (= (csv/cell "  ABC , DEF")
         ["ABC" (seq ", DEF")])))

(deftest test-line
  (is (= (csv/line
"A,B,C
D,E,F"   )
         [["A" "B" "C"] (seq "\nD,E,F")])))

(deftest test-csv
  (is (= (csv/csv
"A,B,C
D,E,F"
         )
         [[["A" "B" "C"] ["D" "E" "F"]]
          ()])))
