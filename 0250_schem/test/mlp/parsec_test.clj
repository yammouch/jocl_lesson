(ns mlp.parsec-test
  (:require [clojure.test :refer :all]
            [mlp.parsec :as psc]
            [clojure.pprint]))

(deftest test-ch1
  (is (= ((psc/ch1 #(= % \A)) "ABC") [\A (seq "BC")])))

(deftest test-all
  (is (= ((psc/all (psc/ch1 #(= % \A))
                   (psc/ch1 #(= % \B)))
          "ABCD")
         [[\A \B]
          (seq "CD")])))

(deftest test-oneof
  (is (= ((psc/oneof (psc/ch1 #{\A}) (psc/ch1 #{\B})) "ABC")
         [\A (seq "BC")]))
  (is (= ((psc/oneof (psc/ch1 #{\A}) (psc/ch1 #{\B})) "BCA")
         [\B (seq "CA")]))
  (is (= ((psc/oneof (psc/ch1 #{\A}) (psc/ch1 #{\B})) "CAB")
         [nil "CAB"])))

(deftest test-times
  (is (= ((psc/times (psc/ch1 #{\A \B}) 0 4) "ABABAB")
         [(seq "ABAB") (seq "AB")]))
  (is (= ((psc/times (psc/ch1 #{\A \B}) 8 nil) "ABABAB")
         [nil ()])))
