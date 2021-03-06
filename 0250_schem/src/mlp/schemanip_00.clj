;(ns mlp.schemanip)
(ns mlp.schemanip
  (:require [clojure.pprint]))

(defn surrounding [y x]
  [[(- y 1)  x    0 (- y 1)  x    0]   ; up
   [   y     x    0 (+ y 1)  x    0]   ; down
   [   y  (- x 1) 1    y  (- x 1) 1]   ; left
   [   y     x    1    y  (+ x 1) 1]]) ; right

(defn nets [y x field]
  [(get-in field [(- y 1) x    0] 0)   ; up
   (get-in field [   y    x    0]  )   ; down
   (get-in field [   y (- x 1) 1] 0)   ; left
   (get-in field [   y    x    1]  )   ; right
   (get-in field [   y    x    2]  )]) ; dot

(defn netd [y x field]
  (mapv vector [:u :d :l :r :c] (nets y x field)))

(defn adjacent [y x d]
  (case d
    :u [(- y 1)  x    0]
    :d [(+ y 1)  x    0]
    :l [   y  (- x 1) 1]
    :r [   y  (+ x 1) 1]))

(defn assoc-net [field y x dir val]
  (case dir
    :u (assoc-in field [(- y 1)  x    0] val)
    :d (assoc-in field [   y     x    0] val)
    :l (assoc-in field [   y  (- x 1) 1] val)
    :r (assoc-in field [   y     x    1] val)))

(defn trace-search-dir [field traced y x d]
  (let [nd          (netd y x field )
        td (into {} (netd y x traced))
        search (filter #(and (#{:u :d :l :r} (% 0)) (= (% 1) 1)) nd)
        search (cond (or (= (get-in nd [4 0]) 1) ; connecting dot
                         (<= (count search) 2))
                     search

                     (= d 0) (filter (comp #{:u :d} first) search)
                     :else   (filter (comp #{:l :r} first) search))
        search (filter #(= (td (% 0)) 1) search)]
    (map first search)))

(defn trace [field y x d]
  (let [cy (count field) cx (count (first field))]
    (loop [stack [[y x d]]
           traced (reduce #(vec (repeat %2 %1)) 0 [2 cx cy])]
      (if (empty? stack)
        traced
        (let [[py px pd] (peek stack)
              search (trace-search-dir field traced py px pd)
              adj (map (partial adjacent py px) search)]
          (recur (into (pop stack)
                       (filter (fn [[sy sx sd]]
                                 (and (< -1 sy cy) (< -1 sx cx)))
                               adj))
                 (reduce #(assoc-net %1 py px %2 1) traced search)
                 ))))))

(defn trace-straight-h [field y x]
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

(defn drawable-h? [y x traced field]
  (cond (= [(get-in field  [y    x    1]  )
            (get-in traced [y    x    1]  )]
           [1 0])
        false
        (= [(get-in field  [y (- x 1) 1] 0)
            (get-in traced [y (- x 1) 1] 0)]
           [1 0])
        false
        (= [(get-in field  [(- y 1) x 0] 0)
            (get-in traced [(- y 1) x 0] 0)
            (get-in field  [   y    x 0]  )
            (get-in traced [   y    x 0]  )]
           [1 0 1 0])
        true
        (= [(get-in field  [(- y 1) x 0] 0)
            (get-in traced [(- y 1) x 0] 0)]
           [1 0])
        false
        (= [(get-in field  [   y    x 0]  )
            (get-in traced [   y    x 0]  )]
           [1 0])
        false
        :else true))

(defn drawable-v? [y x traced field]
  (cond (= [(get-in field  [   y    x 0]  )
            (get-in traced [   y    x 0]  )]
           [1 0])
        false
        (= [(get-in field  [(- y 1) x 0] 0)
            (get-in traced [(- y 1) x 0] 0)]
           [1 0])
        false
        (= [(get-in field  [y (- x 1) 1] 0)
            (get-in traced [y (- x 1) 1] 0)
            (get-in field  [y    x    1]  )
            (get-in traced [y    x    1]  )]
           [1 0 1 0])
        true
        (= [(get-in field  [y (- x 1) 0] 0)
            (get-in traced [y (- x 1) 0] 0)]
           [1 0])
        false
        (= [(get-in field  [y    x    0]  )
            (get-in traced [y    x    0]  )]
           [1 0])
        false
        :else true))

(defn add-dot-h [y x0 x1 traced field]
  (loop [x x0 fld field]
    (if (< x1 x)
      fld
      (recur (+ x 1)
             (if (= (->> (surrounding y x)
                         (filter (fn [[y x d]]
                                   (and (= (get-in fld    [y x d]) 1)
                                        (= (get-in traced [y x d]) 1))))
                         count)
                    3)
               (assoc-in fld [y x 1] 1)
               fld)))))

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
  (when (every? (fn [x] (drawable-h? y x traced field))
                (range x0 (+ x1 1)))
    (->> field
         (draw-net-1-h y x0 x1)
         (add-dot-h y x0 x1 traced))))

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
               (every? (fn [y] (drawable-v? y x traced field))
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
    (let [n (nets y x fld)]
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
