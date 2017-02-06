(ns jocl-lesson.core
  (:gen-class))

(import '(org.jocl CL))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (mlp-cl/init 3)
  (let [{q :queue ctx :context} @mlp-cl/cl-env
        {sub "sub"} @mlp-cl/cl-ker
        {z :z a :a v :v w :w b :b wacc :wacc bacc :bacc} @mlp-cl/cl-mem
        inputs (map (partial cl/create-buffer ctx :f)
                    [[0.0 ] [0.25] [0.75] [1.0 ]])
        labels (map (partial cl/create-buffer ctx :f)
                    (map (partial repeat 3) [0 0 1 1]))]
    (dotimes [i 100]
      (mlp-cl/run-subbatch inputs labels)
      (when (= (mod i 10) 0)
        (let [w (cl/read-float q w 3)
              b (cl/read-float q b 3)]
          (printf "i: %4d w: [%s] b: [%s] err: %8.2f\n"
           i
           (apply str (interpose " " (map (partial format "%6.2f") w)))
           (apply str (interpose " " (map (partial format "%6.2f") b)))
           (mlp-cl/fw-err-subbatch inputs labels)))))
    (doseq [m (concat inputs labels)] (CL/clReleaseMemObject m)))
  (mlp-cl/finalize))
