; lein run -m mlp.gen-sh

(ns mlp.gen-sh
  (:gen-class))

(defn print1 []
  (doseq [[conv-size conv-depth] (for [s [2 5 10 20] d [2 5 10 20]] [s d])]
    (printf (str "java -jar target/uberjar/mlp-0.1.0-SNAPSHOT-standalone.jar  "
                 "10 10 100001 0.1 1 %2d %2d | "
                 "tee result/_10_10_%s_%s.log\n")
            conv-size conv-depth
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-size)))
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-depth))))))

(defn print2 []
  (doseq [[conv-size conv-depth] (for [s [5 10 20 40] d [5 10 20 40]] [s d])]
    (printf (str "java -jar target/uberjar/mlp-0.1.0-SNAPSHOT-standalone.jar  "
                 "20 10 100001 0.1 1 %2d %2d | "
                 "tee result/_20_10_%s_%s.log\n")
            conv-size conv-depth
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-size)))
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-depth))))))

(defn print3 []
  (doseq [[conv-size conv-depth] (for [s [10 20 40 80] d [10 20 40 64]] [s d])]
    (printf (str "java -jar target/uberjar/mlp-0.1.0-SNAPSHOT-standalone.jar  "
                 "100 20 100001 0.1 1 %2d %2d | "
                 "tee result/100_20_%s_%s.log\n")
            conv-size conv-depth
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-size)))
            (apply str (map #(if (= % \space) \_ %)
                            (format "%2d" conv-depth))))))

(defn -main [& _]
  (print1)
  (print2)
  (print3))
