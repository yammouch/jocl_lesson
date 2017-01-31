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
        {sub :sub} @mlp-cl/cl-ker
        {z :z a :a v :v w :w b :b wacc :wacc bacc :bacc
         i0 :i0 i1 :i1 i2 :i2 i3 :i3
         l0 :l0 l1 :l1 l2 :l2 l3 :l3} @mlp-cl/cl-mem]
    (dotimes [_ 500]
      (printf (apply str (repeat 8 "%7s"))
              "in" "w" "b" "z" "a" "v" "wacc" "bacc")
      (newline)
      (mlp-cl/fw i0)
      (mlp-cl/bw i0 l0 true)
      (mlp-cl/fw i1)
      (mlp-cl/bw i1 l1)
      (mlp-cl/fw i2)
      (mlp-cl/bw i2 l2)
      (mlp-cl/fw i3)
      (mlp-cl/bw i3 l3)
      (cl/callk q sub nil [1] :m w :m wacc)
      (cl/callk q sub nil [1] :m b :m bacc);)
      (let [[w] (cl/read-float q w 1)
            [b] (cl/read-float q b 1)]
        (printf "%6.2f %6.2f %6.2f\n" w b (/ (- b) w))))
    ;(clojure.pprint/pprint (cl/read-float q z 1))
    ;(clojure.pprint/pprint (cl/read-float q a 1))
    ;(clojure.pprint/pprint (cl/read-float q v 1))
    (let [[w] (cl/read-float q w 1)
          [b] (cl/read-float q b 1)]
      (printf "%6.2f %6.2f\n" w b))
    ;(clojure.pprint/pprint (cl/read-float q w 1))
    );(clojure.pprint/pprint (cl/read-float q b 1)))
  (mlp-cl/finalize))
