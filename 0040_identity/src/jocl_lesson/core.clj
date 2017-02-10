(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn binary-encode [len n]
  (loop [i len n n acc []]
    (if (<= i 0)
      acc
      (recur (- i 1) (quot n 2) (conj acc (rem n 2)))
      )))

(defn -main [iter intvl len & hidden-layers]
  (let [[iter intvl len] (map read-string [iter intvl len])
        hidden-layers (mapv read-string hidden-layers)
        _ (mlp-cl/init (concat [len] hidden-layers [len]))
        {q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        v (mapv (partial binary-encode len) (range (bit-shift-left 1 len)))
        inputs (mapv (partial cl/create-buffer ctx :f) v)
        labels (mapv (partial cl/create-buffer ctx :f) v)]
    (dotimes [i iter]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i intvl) 0)
        (printf "i: %4d err: %8.2f\n" i
         (mlp-cl/fw-err-subbatch inputs labels))
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
