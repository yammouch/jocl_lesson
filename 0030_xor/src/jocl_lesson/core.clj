(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init [2 3 3 1])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (map (partial cl/create-buffer ctx :f)
                    [[0 0]
                     [0 1]
                     [1 0]
                     [1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0]
                     [1]
                     [1]
                     [0]])]
    (dotimes [i 40001]
    ;(dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 200) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels))
        ;(mlp-cl/dump :w 0)
        ;(mlp-cl/dump :b 0)
        ;(mlp-cl/dump :w 1)
        ;(mlp-cl/dump :b 1)
        ;(mlp-cl/dump :w 2)
        ;(mlp-cl/dump :b 2)
        (flush)
        ))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
