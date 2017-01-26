(ns jocl-lesson.core
  (:gen-class))

(require 'jocl-lesson.mlp-cl)
(alias 'mlp-cl 'jocl-lesson.mlp-cl)

(defn -main [& args]
  (println "Hello, world."))
