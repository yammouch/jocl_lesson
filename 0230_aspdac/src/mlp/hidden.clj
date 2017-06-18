(ns mlp.hidden
  (:gen-class)
  (:require [mlp.mlp-clj :as mlp]
            [clojure.pprint]))

(defn read-file [fname]
  (let [read-data (->> (slurp fname)
                       (#(str "[" % "]"))
                       read-string)]
    [(nth read-data 2) (nth read-data 4)]))

(defn show-l2 [l]
  (let [bunches (->> l
                     (partition 24)
                     (apply map vector)
                     (partition 4)
                     (apply map vector))]
    (doseq [b bunches]
      (doseq [m b]
        (doseq [r (partition 3 m)]
          (->> r
               (map (partial format "%4.1f"))
               (apply println)))
        (newline))
      (newline))))

(defn -main [file-name]
  (let [[l2 l4] (read-file file-name)]
    (show-l2 l2)))
