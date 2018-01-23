(ns mlp.schemedit
 (:require [clojure.pprint])
 (:use [clojure.set :only [difference]]))

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

(defn d-match [[y x] v & fields]
  (->> [:u :d :l :r]
       (filter #(= v (apply net y x % fields)))
       ))

(defn range-2d [end from to o]
  (let [o (case o (:u :d) 0, (:l :r) 1, 0 0, 1 1)
        q (from o)
        to (if (vector? to) (to o) to)]
    (->> (apply range (if (< to q) [(+ q end -1) (- to 1) -1] [q (+ to end)]))
         (map (partial assoc from o)))))

(defn range-p [from to o] (range-2d 1 from to o))
(defn range-n [from to o] (range-2d 0 from to o))

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

(defn beam [field p o]
  (mapv (fn [[d prog]]
          (->> (iterate #(update-in % [o] prog) p)
               (filter (fn [[y x]]
                         (or (= (net y x d  field) [0])    ; net end
                             (= (net y x :f field) [1])))) ; fanout dot
               first))
        [[(case o 0 :u 1 :l) dec]
         [(case o 0 :d 1 :r) inc]]))

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
  (->> (range-p from to os)
       (filter #(= 3 (count (d-match % [1 1] field traced))))
       (reduce (fn [fld [y x]] (assoc-in fld [y x 2] 1))
               field)))

(defn draw-net-1 [from to o field]
  (reduce (fn [fld [y x]] (assoc-in fld [y x o] 1))
          field (range-n from to o)))

(defn stumble [from to o traced field]
  (when (every? (fn [[y x]] (drawable? y x o traced field))
                (range-p from to o))
    [(draw-net-1 from to o traced)
     (as-> field fld
           (draw-net-1 from to o fld)
           (add-dot from to o traced fld))]))

(defn prog [d p]
  (let [[o f] (case d :u [0 dec] :d [0 inc] :l [1 dec] :r [1 inc])]
    (update-in p [o] f)))

(defn search-short [from d traced field]
  (let [cy (- (count field) 1) cx (- (count (first field)) 1)
        [dops to] (case d :u [:d 0] :d [:u cy] :l [:r 0], :r [:l cx])]
    (let [[p] (filter #(as-> % x
                             (d-match x [1 1] traced field)
                             (remove #{dops} x)
                             (not (empty? x)))
                      (range-p from to d))]
      (if p (range-p from p d))
      )))

(defn reach [[y x :as from] d traced field]
  (let [ps (search-short from d traced field)
        o (case d (:u :d) 0 (:l :r) 1)]
    (if (and ps
             (every? (fn [[y x]] (drawable? y x o traced field))
                     ps))
      (let [to (last ps)
            drawn (draw-net-1 from (to o) o field)
            traced-new (draw-net-1 from (to o) o traced)]
        [traced-new
         (if (as-> (d-match to [1 1] traced-new drawn) x
                   (count x)
                   (= 3 x))
           (assoc-in drawn (conj to 2) 1)
           drawn)]))))

(defn debridge [from to o field]
  (as-> field fld
        (reduce #(assoc-in %1 (conj %2 o) 0)
                fld (range-n from to o))
        (reduce #(case (count (d-match %2 [1] %1))
                   (0 1 2) (assoc-in %1 (conj %2 2) 0)
                   3       (assoc-in %1 (conj %2 2) 1)
                   %1)
                fld (range-p from to o))))

(defn shave [from to d field]
  (let [o (case d (:u :d) 0 (:l :r) 1)]
    (loop [[y x :as p] from fld field]
      (let [n (mapcat #(net y x % fld)
                      (case d
                        :u [:u :d :l :r :f]
                        :d [:d :u :l :r :f]
                        :l [:l :r :u :d :f]
                        :r [:r :l :u :d :f]))]
        (if (and (or (= n [1 0 0 0 0])
                     (= n [1 0 1 1 0]))
                 (every? zero? (drop 3 (get-in fld p)))
                 (not= (p o) to))
          (recur (prog d p)
                 (assoc-in fld
                  [(if (= d :u) (- y 1) y)
                   (if (= d :l) (- x 1) x)
                   (case d (:u :d) 0 (:l :r) 1)]
                  0))
          (case (->> (take 4 n)
                     (filter (partial = 1))
                     count)
            (0 1 2) (assoc-in fld [y x 2] 0)
            3       (assoc-in fld [y x 2] 1)
            4       fld))))))

(defn move-element [field [y x :as from] d to]
  (let [to (assoc from d to)
        from-el (drop 3 (get-in field from))
        to-el   (drop 3 (get-in field to  ))]
    (if (every? zero? to-el)
      (as-> field fld
            (update-in fld to
             (fn [v] (vec (concat (take 3 v) from-el))))
            (update-in fld from
             (fn [v] (vec (take (count v)
                                (concat (take 3 v) (repeat 0))
                                ))))))))

(defn move-x [field [y x :as from] to]
  (let [[[y0 _] [y1 _]] (beam field from 0)
        traced (trace field y x 0)
        [d dop] (if (< x to) [:r :l] [:l :r])]
    (as-> field fld
          (move-element fld from 1 to)
          (reach [y0 to] dop traced fld)
          (if (nth fld 1) (apply reach [y1 to] dop fld))
          (if (nth fld 1) (apply stumble [y0 to] y1 0 fld))
          (if (nth fld 1) (debridge [y0 x] y1 0 (nth fld 1)))
          (if fld (shave [y0 x] to d fld))
          (if fld (shave [y1 x] to d fld)))))

(defn move-y [field [y x :as from] to]
  (let [[[_ x0] [_ x1]] (beam field from 1)
        traced (trace field y x 1)
        [d dop] (if (< y to) [:d :u] [:u :d])]
    (as-> field fld
          (move-element fld from 0 to)
          (reach [to x0] dop traced fld)
          (if (nth fld 1) (apply reach [to x1] dop fld))
          (if (nth fld 1) (apply stumble [to x0] x1 1 fld))
          (if (nth fld 1) (debridge [y x0] x1 1 (nth fld 1)))
          (if fld (shave [y x0] to d fld))
          (if fld (shave [y x1] to d fld))
          )))
