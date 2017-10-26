(ns mlp.schemanip)

(defn surrounding [y x]
  [[(- y 1)  x    0 (- y 1)  x    0]   ; up
   [   y     x    0 (+ y 1)  x    0]   ; down
   [   y  (- x 1) 1    y  (- x 1) 1]   ; left
   [   y     x    1    y  (+ x 1) 1]]) ; right

(defn net [y x d & fields]
  (let [idx (case d
              :u [(- y 1) x    0]
              :d [   y    x    0]
              :l [   y (- x 1) 1]
              :r [   y    x    1]
              :f [   y    x    2])]
    (mapv #(get-in % idx 0) fields)))

(defn nets [y x & fields]
  (mapv #(net y x % fields)
        [:u :d :l :r :f]))

(defn nets1 [y x field]
  (vec (apply concat (nets y x field))))

(defn trace-search-dir [field traced y x d]
  (let [search (filter #(= (get-in field (take 3 %) 0) 1)
                       (surrounding y x))
        search (if (or (= (get-in field [y x 2] 0) 1) ; connecting dot
                       (<= (count (filter
                                   #(= (get-in field (take 3 %) 0) 1)
                                   search))
                           2)) ; surrounded by 0, 1, 2 nets
                 search
                 (filter #(= (% 2) d) search))
        search (filter #(= (get-in traced (take 3 %)) 0)
                       search)]
    search))

(defn trace [field y x d]
  (let [cy (count field) cx (count (first field))]
    (loop [stack [[y x d]]
           traced (reduce #(vec (repeat %2 %1)) 0 [2 cx cy])]
      (if (empty? stack)
        traced
        (let [[py px pd] (peek stack)
              search (trace-search-dir field traced py px pd)]
          (recur (into (pop stack)
                       (filter (fn [[sy sx sd]]
                                 (and (< -1 sy cy) (< -1 sx cx)))
                               (map (partial drop 3) search)))
                 (reduce #(assoc-in %1 (take 3 %2) 1) traced search)
                 ))))))

(defn beam-h [field y x]
  [(loop [x x] ; trace left
     (if (or (<= x 0)
             (= (get-in field [y    x    2]) 1)  ; connecting dot
             (= (get-in field [y (- x 1) 1]) 0)) ; net end
       x
       (recur (- x 1))))
   (let [cx (count (first field))]
     (loop [x x] ; trace right
       (if (or (<= cx x)
               (= (get-in field [y    x    2]) 1)  ; connecting dot
               (= (get-in field [y (+ x 1) 1]) 0)) ; net end
         x
         (recur (+ x 1)))))])

(defn drawable? [y x os traced field] ; os:  orientation straight
  (let [dir (case os 0 [:d :u :r :l] 1 [:r :l :d :u])
        [sfwd sbwd ofwd obwd] (map #(net y x % field traced) dir)]
    (cond (=  sfwd        [1 0]       ) false
          (=  sbwd        [1 0]       ) false
          (= [obwd ofwd] [[1 0] [1 0]]) true
          (=  ofwd        [1 0]       ) false
          (=  obwd        [1 0]       ) false
          :else                         true)))

(defn add-dot [from to os traced field]
  (let [fwd #(update-in % [os] inc)
        q0 (from os)
        [from to] (if (< to q0)
                    [(assoc from os to) q0]
                    [       from        to])
        end? #(< to (% os))]
    (loop [[y x :as p] from fld field]
      (if (end? p)
        fld
        (recur (fwd p)
               (if (= (->> [:u :d :r :l]
                           (map #(net y x % fld traced))
                           (filter #(= % [1 1]))
                           count)
                      3)
                 (assoc-in fld [y x 2] 1)
                 fld))))))

(defn draw-net-1-h [y x0 x1 field]
  (loop [x x0 fld field]
    (if (< x0 x)
      fld
      (recur (+ x 1)
             (assoc-in fld [y x 1] 1)))))

(defn draw-net-1-v [y0 y1 x field]
  (loop [y y0 fld field]
    (if (< y0 y)
      fld
      (recur (+ y 1)
             (assoc-in fld [y x 1] 0)))))

(defn stumble-h [y x0 x1 traced field]
  (when (every? (fn [x] (drawable? y x 1 traced field))
                (range x0 (+ x1 1)))
    (->> field
         (draw-net-1-h y x0 x1)
         (add-dot [y x0] x1 1 traced))))

(defn search-short-u [y x traced field]
  (loop [y y]
    (cond (< y 0) false

          (some (fn [[y x d]] (and (= (get-in traced [y x 0] 0) 1)
                                   (= (get-in field  [y x 0] 0) 1)))
                (surrounding y x))
          y

          :else (recur (- y 1))
          )))

(defn reach-u [y x traced field]
  (let [y1 (search-short-u y x traced field)]
    (when (and y1
               (every? (fn [y] (drawable? y x 0 traced field))
                       (range y (- y1 1) -1)))
      (let [drawn (draw-net-1-v y1 (- y 1) x)]
        (if (= (count (filter (fn [[y x d]] (= [(get-in field  [y x d] 0)
                                                (get-in traced [y x d] 0)]
                                               [1 1]))
                              (surrounding y x)))
               2)
          (assoc-in drawn [y1 x 2] 1)
          drawn)))))

(defn debridge-h [y x0 x1 field]
  (reduce #(assoc-in %1 [y %2 2] 0)
          field
          (range x0 x1)))

(defn shave-d [y x field]
  (loop [y y fld field]
    (let [n (nets1 y x fld)]
      (cond (or (= n [0 1 0 0 0])
                (= n [0 1 1 1 0]))
            (recur (+ y 1)
                   (assoc-in fld [y x 0] 0))

            (= (->> (take 4 n)
                    (filter (partial = 1))
                    count)
               3)
            (assoc-in fld [y x 2] 1)

            :else fld))))
