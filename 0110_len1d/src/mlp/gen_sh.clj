; lein run -m mlp.gen-sh

(ns mlp.gen-sh
  (:gen-class))

(defn -main [& _]
  (dorun
    (for [l1 [3 4 5 6 8 10 12 15 20 30]
          l2 [3 4 5 6 8 10 12 15 20 30]]
      (printf "lein run 10 10 100001 0.1 1 %2d %2d | tee result/%s_%s.log\n"
              l1 l2
              (apply str (map #(if (= % \space) \_ %)
                              (format "%2d" l1)))
              (apply str (map #(if (= % \space) \_ %)
                              (format "%2d" l2)))))))
