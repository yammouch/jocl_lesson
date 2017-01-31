(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(require 'clojure.pprint)

(defn -main [& args]
  (mlp-cl/init)
  (let [{q :queue} @mlp-cl/cl-env
        {z :z a :a v :v w :w b :b
         i0 :i0 i1 :i1 i2 :i2 i3 :i3
         l0 :l0 l1 :l1 l2 :l2 l3 :l3} @mlp-cl/cl-mem]
    (mlp-cl/fw i1)
    (mlp-cl/bw i1 l1)
    (clojure.pprint/pprint (cl/read-float q z 1))
    (clojure.pprint/pprint (cl/read-float q a 1))
    (clojure.pprint/pprint (cl/read-float q v 1))
    (clojure.pprint/pprint (cl/read-float q w 1))
    (clojure.pprint/pprint (cl/read-float q b 1)))
  (mlp-cl/finalize))
