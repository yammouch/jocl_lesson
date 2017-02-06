(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init)
  (let [{q :queue} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {z :z a :a v :v w :w b :b wacc :wacc bacc :bacc
         i0 :i0 i1 :i1 i2 :i2 i3 :i3
         l0 :l0 l1 :l1 l2 :l2 l3 :l3} @mlp-cl/cl-mem]
    (dotimes [i 100]
      (mlp-cl/run-subbatch [i0 i1 i2 i3] [l0 l1 l2 l3])
      (when (= (mod i 10) 0)
        (let [w (cl/read-float q w 3)
              b (cl/read-float q b 3)]
          (printf "i: %4d w: [%s] b: [%s] err: %8.2f\n"
           i
           (apply str (interpose " " (map (partial format "%6.2f") w)))
           (apply str (interpose " " (map (partial format "%6.2f") b)))
           (mlp-cl/fw-err-subbatch [i0 i1 i2 i3] [l0 l1 l2 l3])
           )))))
  (mlp-cl/finalize))
