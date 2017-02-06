(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init 3 4)
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {z :z a :a v :v w :w b :b wacc :wacc bacc :bacc} @mlp-cl/cl-mem
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
                    [[0 0 0 0]
                     [0 0 1 0]
                     [0 1 0 0]
                     [0 1 1 1]
                     [1 0 0 0]
                     [1 0 1 1]
                     [1 1 0 1]
                     [1 1 1 1]])]
    (dotimes [i 200]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 10) 0)
        (let [w (cl/read-float q w 12)
              b (cl/read-float q b 4)]
          (printf "i: %4d b: [%s] err: %8.2f w: (follows)\n"
           i
           (apply str (interpose " " (map (partial format "%6.2f") b)))
           (mlp-cl/fw-err-subbatch inputs labels))
          (doseq [row (partition 4 w)]
            (apply println (map (partial format "%6.2f") row))
            ))))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
