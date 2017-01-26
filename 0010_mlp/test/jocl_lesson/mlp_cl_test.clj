(ns jocl-lesson.mlp-cl-test
  (:require [clojure.test :refer :all]
            [jocl-lesson.mlp-cl :as mlp-cl]
            [jocl-lesson.cl     :as cl    ]
            [clojure.pprint               ])
  (:import  [org.jocl Sizeof]))

(deftest a-test
  (mlp-cl/init)
  (let [n 4
        {q :queue} @mlp-cl/cl-env
        {set0 :set0} @mlp-cl/cl-ker
        mem (cl/create-buffer (@mlp-cl/cl-env :context)
                              (* n Sizeof/cl_float))]
    (cl/callk q set0 nil [n] :m mem)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem n) (repeat n 0))
                )))
  (mlp-cl/finalize)
  (is (= 1 1)))
