(ns mlp.schemanip)

(defn surrounding [y x]
  (filter (fn [[ly lx]]
            (and (<= 0 lx) (<= 0 ly)
            (= (get-in traced [ly lx]) 0)))
    [[(- y 1)  x    0 (- y 1)  x    0]   ; up
     [   y     x    0 (+ y 1)  x    0]   ; down
     [   y  (- x 1) 1    y  (- x 1) 1]   ; left
     [   y     x    1    y  (+ x 1) 1]]) ; right

(defn trace-net [field y x d]
  (let [cy (count field) cx (count (first field))]
    (loop [[[py px pd] :as stack] [[y x d]]
           traced (reduce #(repeat %2 %1) 0 [2 cx cy])]
      (if (empty? stack)
        traced
        (let [search (surrounding py px)
              search (if (or (= (get-in field [y x 2]) 1) ; connecting dot
                             (<= (count (filter
                                         #(= (get-in field (take 3 %)) 1)
                                         search)))
                                 2)) ; surrounded by 0, 1, 2 nets
                       search
                       (filter #(= (% 2) d) search))]
          (recur (into (pop stack)
                       (filter (fn [[_ _ _ sy sx sd]]
                                 (and (< -1 sy cy) (< -1 sx cx)))
                               search))
                 (reduce #(assoc-in traced % 1) search)
                 ))))))

(defn trace-straight-h [field y x]
  [(loop [x x] ; trace left
     (if (or (<= x 0)
             (= (get-in field [y    x    2]) 1)  ; connecting dot
             (= (get-in field [y (- x 1) 1]) 0)) ; net end
       x
       (loop (- x 1)))
   (let [cx (count (first field))]
     (loop [x x] ; trace right
       (if (or (<= cx x)
               (= (get-in field [y    x    2]) 1)  ; connecting dot
               (= (get-in field [y (+ x 1) 1]) 0)) ; net end
       x
       (loop (+ x 1))))])

(defn surrounding-h [y x0 x1]
  (concat
   (when (< 0 y)
     (map (fn [x] [(- y 1) x 0])
          (range x0 (+ x1 1))))
   (when (< x0 0) [[y (- x 1) 1]])
   (map (fn [x] [y x 0])
        (range x0 (+ x1 1)))
   (map (fn [x] [y x 1])
        (range x0 (+ x1 1))
        )))

(defn disturbance-h [y x0 x1 traced field]
  (some (fn [[y x d]]
          (and (= (get-in field  [y x d]) 1)
               (= (get-in traced [y x d]) 0)))
        (surrounding-h y x0 x1)))

(defn add-dot-h [y x0 x1 traced field]
  (loop [x x0 fld field]
    (if (< x1 x)
      fld
      (recur (+ x 1)
             (if (= (->> (surrounding y x)
                         (filter (fn [[y x d]]
                                   (and (= (get-in fld    [y x d]) 1)
                                        (= (get-in traced [y x d]) 1)))
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

(defn draw-net [y x0 x1 traced field]
  (when-not (disturbance-h y x0 x1 traced field)
    (->> field
         (draw-net-1-h y x0 x1)
         (add-dot-h y x0 x1 traced))))
