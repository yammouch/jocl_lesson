(ns jocl-lesson.mlp-cl-test
  (:require [clojure.test :refer :all]
            [jocl-lesson.mlp-cl :as mlp-cl]
            [jocl-lesson.cl     :as cl    ]
            [clojure.pprint               ])
  (:import  [org.jocl CL Sizeof Pointer]))

(deftest set0-test
  (mlp-cl/init)
  (let [n 4
        {q :queue} @mlp-cl/cl-env
        {set0 :set0} @mlp-cl/cl-ker
        mem (cl/create-buffer (@mlp-cl/cl-env :context) :f n)]
    (cl/callk q set0 nil [n] :m mem)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem n) (repeat n 0))))
    (CL/clReleaseMemObject mem))
  (mlp-cl/finalize))

(deftest dense-fw-test
  (mlp-cl/init)
  (let [{q :queue} @mlp-cl/cl-env
        {k :dense-fw} @mlp-cl/cl-ker
        w 4, h 3
        [mem-m mem-in mem-out :as mems]
        (map (fn [n]
               (cl/create-buffer (@mlp-cl/cl-env :context) :f n))
             [(* w h) h w])
        in [3 2 1]
        m  [ 1  2  3  4
             2  4  6  8
             3  6  9 12]]
    (CL/clEnqueueWriteBuffer q mem-in CL/CL_TRUE
     0 (* h Sizeof/cl_float) (Pointer/to (float-array in))
     0 nil nil)
    (CL/clEnqueueWriteBuffer q mem-m CL/CL_TRUE
     0 (* h w Sizeof/cl_float) (Pointer/to (float-array m))
     0 nil nil)
    (cl/callk q k nil [w] :m mem-out :m mem-in :m mem-m :i w :i h)
    (is (every? #(< -0.01 % 0.01)
                (map - (cl/read-float q mem-out w)
                       [10 20 30 40])))
    (doseq [m mems] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
