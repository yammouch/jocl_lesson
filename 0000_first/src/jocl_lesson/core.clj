(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.fft-cl)
(alias 'fft 'jocl-lesson.fft-cl)

(require 'clojure.pprint)

(defn -main [& args]
  (dosync (ref-set fft/exp2 4))
  (fft/init)
  (let [n (bit-shift-left 1 @fft/exp2)
        phase (/ (* 2 Math/PI) n)
        swing-0db (bit-shift-left 1 14)]
    (clojure.pprint/pprint
     (fft/fft-mag-norm
      (float-array (map #(Math/cos (* phase %)) 
                        (range n)))
      0 1.0)))
  (fft/finalize))
