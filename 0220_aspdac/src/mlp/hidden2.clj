(ns mlp.hidden2
  (:gen-class))

(defn read-file []
  (->> (slurp "data/p001.dat")
       (#(str "[" % "]"))
       read-string
       (#(nth % 4))))

(defn -main [& args]
  (->> (read-file)
       (take 4)
       println))
