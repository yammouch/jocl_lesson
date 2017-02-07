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
    (dotimes [i 10001]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 500) 0)
        (let [w0 (cl/read-float q (w 0) 12)
              b0 (cl/read-float q (b 0) 4)
              w1 (cl/read-float q (w 1) 20)
              b1 (cl/read-float q (b 1) 5)]
          (printf "i: %4d err: %8.2f\n"
           i
           (mlp-cl/fw-err-subbatch inputs labels))
          (printf "b0: %s, w0 follows\n"
           (apply str (interpose " " (map (partial format "%6.2f") b0))))
          (doseq [row (partition 4 w0)]
            (apply println (map (partial format "%6.2f") row)))
          (printf "b1: %s, w1 follows\n"
           (apply str (interpose " " (map (partial format "%6.2f") b1))))
          (doseq [row (partition 5 w1)]
            (apply println (map (partial format "%6.2f") row))
            ))))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
