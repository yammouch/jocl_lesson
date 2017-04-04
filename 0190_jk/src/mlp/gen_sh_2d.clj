; lein run -m mlp.gen-sh

(ns mlp.gen-sh-2d
  (:gen-class))

(defn print1 []
  (doseq [[conv-size conv-depth]
           (for [s [6 7 8 10] d [10 11 12 13 15 17 20]] [s d])]
    (printf (str "lein run -m mlp.len2d "
                 " 20 10 200001 0.1 1 %2d %2d | "
                 "tee result_len2d/_20_10_%s_%s.log\n")
            conv-size conv-depth
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-size)))
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-depth))))))

(defn -main [& _]
  (print1))
