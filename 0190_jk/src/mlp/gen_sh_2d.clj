; lein run -m mlp.gen-sh

(ns mlp.gen-sh-2d
  (:gen-class))

(defn print1 []
  (doseq [[conv-size conv-depth]
          (->> (for [s [10 8 7 6] d [20 17 15 13 12 11 10]] [s d])
               (sort-by (fn [[s d]] (* s s d)))
               reverse)]
    (printf (str "lein run -m mlp.len2d "
                 " 20 10 100001 0.1 1 %2d %2d | "
                 "tee result_len2d/_20_10_%s_%s.log\n")
            conv-size conv-depth
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-size)))
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-depth))))))

(defn -main [& _]
  (print1))
