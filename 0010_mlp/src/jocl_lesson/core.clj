(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(require 'clojure.pprint)

(defn -main [& args]
  (mlp-cl/init)
  (println (-> (@mlp-cl/cl-ker :add)
               (mlp-cl/clGetKernelInfo 'CL_KERNEL_FUNCTION_NAME)
               (cl/parse-str-info)))
  (let [{q :queue} @mlp-cl/cl-env
        {sub :sub} @mlp-cl/cl-ker
        {z :z a :a v :v w :w b :b wacc :wacc bacc :bacc
         i0 :i0 i1 :i1 i2 :i2 i3 :i3
         l0 :l0 l1 :l1 l2 :l2 l3 :l3} @mlp-cl/cl-mem]
    (dotimes [i 100]
      (mlp-cl/run-subbatch [i0 i1 i2 i3] [l0 l1 l2 l3])
      (when (= (mod i 10) 0)
        (let [[w] (cl/read-float q w 1)
              [b] (cl/read-float q b 1)]
          (printf "i: %4d w: %6.2f b: %6.2f -b/w: %6.2f\n" i w b (/ (- b) w))
          ))))
  (mlp-cl/finalize))
