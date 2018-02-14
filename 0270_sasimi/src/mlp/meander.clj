(ns mlp.meander
  (:require [mlp.util :as utl]
            [mlp.schemprep :as scp]
            [clojure.pprint]))

(defn range-2d [end from to o]
  (let [o (case o (:u :d) 0, (:l :r) 1, 0 0, 1 1)
        q (from o)
        to (if (vector? to) (to o) to)]
    (->> (apply range (if (< to q) [(+ q end -1) (- to 1) -1] [q (+ to end)]))
         (map (partial assoc from o)))))

(defn range-n [from to o] (range-2d 0 from to o))

(defn line [field from to o]
  (reduce (fn [fld [y x]] (assoc-in fld [y x o] 1))
          field (range-n from to o)))

;    |<-  l0  ->|
;   _            p1
;  |_>----------+  -
;    p0         |  ^
;               |  l1
;     p3      p2|  v
;   -  +--------+  -
;   ^  |<- l2 ->|
;  l3  |
;   v  |p4       |\  p6       _
;   -  +---------| >o--------|_>
;              p5|/         p7
;
;      |<- l4  ->|  |<- l5 ->|

(defn meander-0-0 [l]
  (let [p0 [0 0]
        p1 (update-in p0 [1] + (l 0))
        p2 (update-in p1 [0] + (l 1))
        p3 (update-in p2 [1] - (l 2))
        p4 (update-in p3 [0] + (l 3))
        p5 (update-in p4 [1] + (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 14 14]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (line fld p1 (p2 0) 0)
          (line fld p2 (p3 1) 1)
          (line fld p3 (p4 0) 0)
          (line fld p4 (p5 1) 1)
          (assoc-in fld (conj p5 5) 1)  ; not
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-x
          :org [(quot (+ (p1 0) (p2 0)) 2)
                (p1 1)]
          :dst (p3 1)}}))

(defn meander-0-1 [l]
  (let [p0 [0 0]
        p1 (update-in p0 [1] + (- (l 0) (l 2)))
        p4 (update-in p1 [0] + (+ (l 1) (l 3)))
        p5 (update-in p4 [1] + (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 14 14]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (line fld p1 (p4 0) 0)
          (line fld p4 (p5 1) 1)
          (assoc-in fld (conj p5 5) 1)  ; not
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-y
          :org p0
          :dst (p4 0)}}))

(def meander-0 (juxt meander-0-0 meander-0-1))

(defn meander-pos [n]
  (let [m (vec (meander-0 [4 2 2 2 4 2]))
        [u d l r] (scp/room (get-in m [0 :field]))
        ml (for [dy (range (- u) (+ d 1)) dx (range (- l) (+ r 1))]
             [dy dx])
        [ml] (utl/select (vec ml) [n] (utl/xorshift 2 4 6 8))]
    (mapv (fn [mv] 
            (mapv (fn [pair]
                    (as-> pair p
                          (update-in p [:field     ] scp/slide mv)
                          (update-in p [:cmd :org 0] + (mv 0))
                          (update-in p [:cmd :org 1] + (mv 1))
                          (update-in p [:cmd :dst  ] +
                                     (mv (case (get-in pair [:cmd :cmd])
                                           :move-y 0 :move-x 1)))))
                  m))
          ml)))

(defn -main []
  (as-> (meander-pos 4) x
        (doseq [pos x]
          (doseq [sequ pos]
            (clojure.pprint/pprint
             (scp/format-field (:field sequ)))
            (clojure.pprint/pprint (:cmd sequ))
            ))))
