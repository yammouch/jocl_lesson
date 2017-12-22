(ns mlp.parsec-test
  (:require [clojure.test :refer :all]
            [mlp.parsec :as psc]
            [clojure.pprint]))

(deftest test-char
  (is (= ((psc/char #(= % \A)) "ABC") [\A (seq "BC")])))

(deftest test-all
  (is (= ((psc/all (psc/char #(= % \A))
                   (psc/char #(= % \B)))
          "ABCD")
         [[\A \B]
          (seq "CD")])))

(deftest test-times
  (is (= ((psc/times (psc/char #{\A \B}) 0 4) "ABABAB")
         [(seq "ABAB") (seq "AB")]))
  (is (= ((psc/times (psc/char #{\A \B}) 8 nil) "ABABAB")
         [nil ()])))
