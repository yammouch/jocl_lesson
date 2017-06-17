(ns mlp.mlp-clj
  (:gen-class))

(defn softmax [v]
  (let [max-v (apply max v)
        exp-v (map #(Math/exp (- % max-v)) v)
        sum (apply + exp-v)]
    (map #(/ % sum) exp-v)))

(defn dimension [x]
  (if (coll? x)
    (conj (dimension (first x)) (count x))
    []))

(defn padding [m pu pd pl pr]
  (let [w (+ pl pr (count (first m)))
        zero (reduce #(repeat %2 %1) 0.0 (dimension (ffirst m)))]
    (concat (repeat pu (repeat w zero))
            (map #(concat (repeat pl zero) % (repeat pr zero)) m)
            (repeat pd (repeat w zero)))))

; (* [i0 i1 i2 i3]
;    [[c00 c01 c02]
;     [c10 c11 c12]
;     [c20 c21 c22]
;     [c30 c31 c32]])
; = [(+ (* i0 c00) (* i1 c10) (* i2 c20) (* i3 c30))
;    (+ (* i0 c01) (* i1 c11) (* i2 c21) (* i3 c31))
;    (+ (* i0 c02) (* i1 c12) (* i2 c22) (* i3 c32))]
; (* [i0 i1 i2] [c0 c1 c2])
; = [[(* i0 c0) (* i0 c1) (* i0 c2)]
;    [(* i1 c0) (* i1 c1) (* i1 c2)]
;    [(* i2 c0) (* i2 c1) (* i2 c2)]]

(defn shape [x]
  (cond (not (coll?        x )) :scalar
        (not (coll? (first x))) :vector
        :else                   :matrix))

(defn *c [x0 x1]
  (case (map shape [x0 x1])
    [:scalar :scalar] (* x0 x1)
    [:scalar :vector] (map (partial * x0) x1)
    [:vector :scalar] (map (partial * x1) x0)
    [:vector :vector] (map (fn [e0]
                             (map (fn [e1] (* e0 e1))
                                  x1))
                           x0)
    [:vector :matrix] (apply map (fn [& v] (apply + (map * x0 v))) x1)
    [:matrix :vector] (      map (fn [  v] (apply + (map * v x1))) x0)
    ))

(defn +r [x0 x1]
  (cond (every? (comp not coll?) [x0 x1])
        (if (every? nil? [x0 x1]) [] (+ x0 x1))

        (every? coll? [x0 x1]) (cons (+r (first x0) (first x1))
                                     (+r (next  x0) (next  x1))
                                     )))

(defn conv-cell [i c bw?]
  (let [[x0 x1] (if bw? [c i] [i c])]
    (reduce +r (map *c (apply concat x0) (apply concat x1)))
    ))

(defn conv-fw
 ([i c] (conv-fw i c false))
 ([i c bw?]
  (let [ch (count c)
        cw (count (first c))]
    (map (fn [rows]
           (apply map (fn [& vs] (conv-cell vs c bw?))
                      (map (partial partition cw 1) rows)))
         (partition ch 1 i)))))
