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
          (recur (into stack (filter (fn [[_ _ _ sy sx sd]]
                                       (and (< -1 sy cy) (< -1 sx cx)))
                                     search))
                 (recude #(assoc-in traced % 1) search)
                 ))))))

