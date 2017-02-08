(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init [3 4 5])
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {w :w b :b} @mlp-cl/cl-mem
        inputs (map (partial cl/create-buffer ctx :f)
                    [[0 0 0]
                     [0 0 1]
                     [0 1 0]
                     [0 1 1]
                     [1 0 0]
                     [1 0 1]
                     [1 1 0]
                     [1 1 1]])
        labels (map (partial cl/create-buffer ctx :f)
                    [[0 0 0 0 1]
                     [0 0 1 0 1]
                     [0 1 0 0 1]
                     [0 1 1 1 1]
                     [1 0 0 0 1]
                     [1 0 1 1 1]
                     [1 1 0 1 1]
                     [1 1 1 1 0]])]
    ;(dotimes [i 10001]
    (dotimes [i 1]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 500) 0)
        (printf "i: %4d err: %8.2f\n"
         i
         (mlp-cl/fw-err-subbatch inputs labels)
         )))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
