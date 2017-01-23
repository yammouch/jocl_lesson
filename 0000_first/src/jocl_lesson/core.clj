(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.fft-cl)
(alias 'fft-cl 'jocl-lesson.fft-cl)

(require 'jocl-lesson.fft)
(alias 'fft 'jocl-lesson.fft)

(require 'clojure.pprint)

(defn -main [& args]
  (dosync (ref-set fft-cl/exp2 4))
  (fft-cl/init)
  (let [n (bit-shift-left 1 @fft-cl/exp2)
        phase (/ (* 2 Math/PI) n)
        swing-0db (bit-shift-left 1 14)
        input (map #(Math/cos (* phase %))
                   (range n))
        result (vec (fft/fft-mag-norm input 1.0))
        ;result (update-in result [0] inc)
        result-cl (fft-cl/fft-mag-norm (float-array input) 0 1.0)]
    (clojure.pprint/pprint result)
    (clojure.pprint/pprint result-cl)
    (if (every? #(< -0.01 % 0.01) (map - result result-cl))
      (println "[OK]")
      (println "!ER!")
      ))
  (fft-cl/finalize))
