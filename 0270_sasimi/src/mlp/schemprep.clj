(ns mlp.schemprep)

(defn count-empty-row-up [field]
  (as-> field fld
        (take-while (partial every? (partial every? zero?)) fld)
        (count fld)))

(defn room [field]
  (let [b (reverse field)
        l (apply map vector field)
        r (reverse l)]
    (map count-empty-row-up [field b l r])))
