; lein run -m mlp.t0150-gentr

(ns mlp.t0150-gentr
  (:gen-class)
  (:require [clojure.pprint]
            [mlp.util :as utl]
            [mlp.schemmlp]
            [mlp.schemprep :as spp]
            [mlp.meander]))

(defn position-variation [m rs]
  (let [m (vec m)
        [u d l r] (spp/room (get-in m [0 :field]))
        ml (for [dy (range (- u) (+ d 1)) dx (range (- l) (+ r 1))]
             [dy dx])
        n (count ml)
        [mtr mts] (utl/select (vec ml) [(- n 1) 1] rs)]
    ;[(mapv (partial mlp.schemmlp/slide-history m) mtr)
    [(mapv (partial mlp.schemmlp/slide-history m) (vec ml))
     (mapv (partial mlp.schemmlp/slide-history m) mts)]))

(defn meander-0-geometry-variation [size]
  (for [g0 [2]
        [g1 g3] (concat (for [g1 [ 2  3  4] g3 [ 2]] [g1 g3])
                        (for [g1 [-2 -3 -4] g3 [-2]] [g1 g3]))
        g2 [2 3 4] g4 [2] g5 [2]]
    (mlp.meander/meander-0 size [(+ g0 g2) g1 g2 g3 g4 g5])))

(defn ring-0-geometry-variation [size]
  (for [g0 [4]
        [g1 g2] (concat (for [g1 [ 4] g2 [ 2  3  4]] [g1 g2])
                        (for [g1 [-4] g2 [-2 -3 -4]] [g1 g2]))
        g3 [2 3 4] g4 [1] g5 [2]]
    (mlp.meander/ring-0 size [g0 g1 g2 g3 g4 g5])))

(defn ring-1-geometry-variation [size]
  (for [g0 [2] g1 [1]
        [g2 g3] (concat (for [g2 [ 2] g3 [ 2  3  4]] [g2 g3])
                        (for [g2 [-2] g3 [-2 -3 -4]] [g2 g3]))
        g4 [2 3 4] g5 [2]]
    (mlp.meander/ring-1 size [g0 g1 g2 g3 g4 g5])))

(defn test-pattern [size]
  (as-> (concat ;(meander-0-geometry-variation size)
                (ring-0-geometry-variation size)
                (ring-1-geometry-variation size)) s
        ; [h h ...]
        (mapv (fn [geom]
                (position-variation geom (utl/xorshift 2 4 6 8)))
              s)
        ; [ [[h h ...] [h h ...]]
        ;   [[h h ...] [h h ...]]
        ;   ... ]

        (apply map vector s)
        ; [ [[h h ...] [h h ...] ...]
        ;   [[h h ...] [h h ...] ...] ]
        ))

(defn -main []
  (let [height 10, width 10
        p (test-pattern [height width])
        [tr ts] (map (comp (partial apply concat)
                           (partial apply concat))
                     p)
        tr (mapv #(mlp.schemmlp/make-input-label % height width) tr)
        ts (mapv #(mlp.schemmlp/make-input-label % height width) ts)]
    (clojure.pprint/pprint tr)
    (clojure.pprint/pprint ts)))
