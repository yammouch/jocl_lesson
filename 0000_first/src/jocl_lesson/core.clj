(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.fft-cl)
(alias 'fft 'jocl-lesson.fft-cl)

(require 'clojure.pprint)

(defn -main [& args]
  (fft/init)
  (clojure.pprint/pprint
   (fft/fft-mag-norm (byte-array [0 0 1 0 0 0 -1 -1]) 0 1.0))
  (fft/finalize))
