(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init 3 5)
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
    (dotimes [i 1001]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 50) 0)
        (let [w (cl/read-float q (w 0) 15)
              b (cl/read-float q (b 0) 5)]
          (printf "i: %4d b: [%s] err: %8.2f w: (follows)\n"
           i
           (apply str (interpose " " (map (partial format "%6.2f") b)))
           (mlp-cl/fw-err-subbatch inputs labels))
          (doseq [row (partition 5 w)]
            (apply println (map (partial format "%6.2f") row))
            ))))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
