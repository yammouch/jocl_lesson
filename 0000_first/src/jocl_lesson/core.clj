(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.fft-cl)
(alias 'fft-cl 'jocl-lesson.fft-cl)

(require 'jocl-lesson.fft)
(alias 'fft 'jocl-lesson.fft)

(require 'clojure.pprint)

(defn let-err-test []
  (macroexpand-1
   `(fft-cl/let-err err
      [len (bit-shift-left 1 exp2)
       w-mem (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
              (* len Sizeof/cl_float) nil err)
       buf0 (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
             (* len 2 Sizeof/cl_float) nil err)]
      {:w w-mem :buf0 buf0})))

(defn -main [& args]
  (clojure.pprint/pprint (let-err-test))
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
