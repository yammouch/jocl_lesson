(ns mlp.field
  (:gen-class))

(defn start-stops [fs ml] ; fs: field-size, ml: max-len
  (mapcat (fn [start]
            (map (fn [stop] [start stop])
                 (range start (min fs (+ start ml)))))
          (range fs)))

(defn field1 [fs start stop q dir]
  (reduce (case dir
            :h (fn [fld s] (assoc-in fld [q s] 1))
            :v (fn [fld s] (assoc-in fld [s q] 1))
            identity)
          (reduce (fn [acc x] (vec (repeat x acc)))
                  0 [fs fs])
          (range start (inc stop))))

(defn one-hot [fs i]
  (assoc (vec (repeat fs 0)) (dec i) 1))
