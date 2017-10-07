(ns mlp.schemanip-test
  (:require [clojure.test :refer :all]
            [mlp.schemanip :as smp]))

(deftest test-surrouding
  (is (= (smp/surrounding 4 6)
         [[3 6 0 3 6 0]
          [4 6 0 5 6 0]
          [4 5 1 4 5 1]
          [4 6 1 4 7 1]])))
