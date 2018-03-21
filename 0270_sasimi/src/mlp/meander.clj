(ns mlp.meander
  (:require [mlp.util :as utl]
            [mlp.schemprep :as smp]
            [mlp.schemmlp  :as scp]
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

(defn meander-0-0 [[h w] l]
  (let [y0 (+ (l 1) (l 3))
        p0 [(if (< y0 0) (- y0) 0) 0]
        p1 (update-in p0 [1] + (l 0))
        p2 (update-in p1 [0] + (l 1))
        p3 (update-in p2 [1] - (l 2))
        p4 (update-in p3 [0] + (l 3))
        p5 (update-in p4 [1] + (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
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

(defn meander-0-1 [[h w] l]
  (let [y0 (+ (l 1) (l 3))
        p0 [(if (< y0 0) (- y0) 0) 0]
        p1 (update-in p0 [1] + (- (l 0) (l 2)))
        p4 (update-in p1 [0] + (+ (l 1) (l 3)))
        p5 (update-in p4 [1] + (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
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
  (let [m (vec (meander-0 [14 14] [4 2 2 2 4 2]))
        [u d l r] (smp/room (get-in m [0 :field]))
        _ (println [u d l r])
        ml (for [dy (range (- u) (+ d 1)) dx (range (- l) (+ r 1))]
             [dy dx])
        [ml] (utl/select (vec ml) [n] (utl/xorshift 2 4 6 8))]
    (mapv (partial scp/slide-history m) ml)))

;    |<-  l0  ->|
;   _            p1
;  |_>----------+  -
;    p0         |  ^
;               |  l1
;     p4        |  v     |\            _
;      +--------+--------| >o---------|_>
;      |<- l3 ->|  ^   p5|/  p6     p7
;      |        |  l2
;      |p3    p2|  v
;      +--------+
;                <- l4 ->    <- l5 ->

(defn ring-0-0 [[h w] l]
  (let [y0 (+ (l 1) (l 2))
        x0 (- (l 0) (l 3))
        p0 [(if (< y0 0) (- y0) 0)
            (if (< x0 0) (- x0) 0)]
        p1 (update-in p0 [1] + (l 0))
        p2 (update-in p1 [0] + (l 1) (l 2))
        p3 (update-in p2 [1] - (l 3))
        p4 (update-in p3 [0] - (l 2))
        p5 (update-in p4 [1] + (l 3) (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (line fld p1 (p2 0) 0)
          (line fld p2 (p3 1) 1)
          (line fld p3 (p4 0) 0)
          (line fld p4 (p5 1) 1)
          (assoc-in fld (conj p5 5) 1)  ; not
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-y
          :org [(p2 0)
                (quot (+ (p2 1) (p3 1)) 2)]
          :dst (p4 0)}}))

(defn ring-0-1 [[h w] l]
  (let [y0 (+ (l 1) (l 2))
        x0 (- (l 0) (l 3))
        p0 [(if (< y0 0) (- y0) 0)
            (if (< x0 0) (- x0) 0)]
        p1 (update-in p0 [1] + (l 0))
        p4 (mapv + p1 [(l 1) (- (l 3))])
        p5 (update-in p4 [1] + (l 3) (l 4))
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (line fld p1 (p4 0) 0)
          (line fld p4 (p5 1) 1)
          (assoc-in fld [(p4 0) (p1 1) 2] 1) ; fanout point
          (assoc-in fld (conj p5 5) 1)  ; not
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-x
          :org p4
          :dst (p1 1)}}))

(defn ring-0-2 [[h w] l]
  (let [y0 (+ (l 1) (l 2))
        x0 (- (l 0) (l 3))
        p0 [(if (< y0 0) (- y0) 0)
            (if (< x0 0) (- x0) 0)]
        p1 (update-in p0 [1] + (l 0))
        p5 (mapv + p1 [(l 1) (l 4)])
        p6 (update-in p5 [1] + 2)
        p7 (update-in p6 [1] + (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (line fld p1 (p5 0) 0)
          (line fld [(p5 0) (p1 1)] (p5 1) 1)
          (assoc-in fld (conj p5 5) 1)  ; not
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-y
          :org p0
          :dst (p5 0)}}))

(def ring-0 (juxt ring-0-0 ring-0-1 ring-0-2))

;    |<-  l0  ->|  |<- l1 ->|
;
;   _           |\         p3
;  |_>----------| >o--------+  -
;     p0      p1|/  p2      |  ^
;                           |  l2
;                           |  v   p7 _
;                  p6+------+--------|_>
;                    |      |  ^
;                    |      |  l3
;                    |    p4|  v
;                  p5+------+  -
;
;                    |<-l4->|<- l5 ->|

(defn ring-1-0 [[h w] l]
  (let [y0 (+ (l 2) (l 3))
        x0 (- (l 4) (l 0) 2 (l 1))
        p0 [(if (< y0 0) (- y0) 0)
            (if (< 0 x0) x0 0)]
        p1 (update-in p0 [1] + (l 0))
        p2 (update-in p1 [1] + 2)
        p3 (update-in p2 [1] + (l 1))
        p4 (update-in p3 [0] + (l 2) (l 3))
        p5 (update-in p4 [1] - (l 4))
        p6 (update-in p5 [0] - (l 3))
        p7 (update-in p5 [1] + (l 4) (l 5))]
   {:field
    (as-> (reduce #(vec (repeat %2 %1)) 0 [6 w h]) fld
          (assoc-in fld (conj p0 3) 1)  ; in
          (line fld p0 (p1 1) 1)
          (assoc-in fld (conj p1 5) 1)  ; not
          (line fld p2 (p3 1) 1)
          (line fld p3 (p4 0) 0)
          (line fld p4 (p5 1) 1)
          (line fld p5 (p6 0) 0)
          (line fld p6 (p7 1) 1)
          (assoc-in fld (conj p7 4) 1)) ; out
    :cmd {:cmd :move-y
          :org [(p4 0)
                (quot (+ (p4 1) (p5 1)) 2)]
          :dst (p6 0)}}))

(defn -main []
  ;(doseq [sequ (ring-0 [14 14] [4 -2 -3 3 2 2])]
  (doseq [sequ (meander-0 [14 14] [4 2 2 2 4 2])]
    (clojure.pprint/pprint
     (smp/format-field (:field sequ)))
    (clojure.pprint/pprint (:cmd sequ))
    ))
