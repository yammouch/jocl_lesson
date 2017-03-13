(ns mlp.gen-sh
  (:gen-class))

(defn -main [& _]
  (dorun
    (for [lr [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.3] ; learning rate
          seed (range 1 11)]
      (printf "lein run 10 10 100001 %5.3f %2d 30 30 | tee result/%s_%s.log\n"
              lr seed
              (apply str (map #(if (= % \.) \R %)
                              (format "%5.3f" lr)))
              (apply str (map #(if (= % \space) \_ %)
                              (format "%2d" seed)))))))
