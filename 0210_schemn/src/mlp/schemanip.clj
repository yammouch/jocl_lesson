(ns mlp.schemanip)

; {:cmd :move-x :org [4 6] :dst 8}

(defn slide-upper-field [field empty]
  (assoc field :body
         (concat (next (field :body))
                 [(repeat (count (nth (field :body) 0)) empty)]
                 )))

(defn slide-upper
 ([x] (slide-upper x 0))
 ([{field :field cmd :cmd :as x} empty]
  (if (or (= (get-in cmd [:org 1]) 0)
          (and (= (:cmd cmd) :move-y)
               (= (:dst cmd) 0))
          (some (partial not= empty)
                (first (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-y)
          (update-in x [:cmd :dst] dec)
          x)
        (update-in [:field] #(slide-upper-field % empty))
        (update-in [:cmd :org 1] dec)))))

(defn slide-lower-field [field empty]
  (assoc field :body
         (cons (repeat (count (nth (field :body) 0)) empty)
               (butlast (field :body)))))

(defn slide-lower
 ([x] (slide-lower x 0))
 ([{field :field cmd :cmd :as x} empty]
  (if (or (<= (count (:body field)) (get-in cmd [:org 1]))
          (and (= (:cmd cmd) :move-y)
               (<= (count (:body field)) (:dst cmd)))
          (some (partial not= empty)
                (last (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-y)
          (update-in x [:cmd :dst] inc)
          x)
        (update-in [:field] #(slide-lower-field % empty))
        (update-in [:cmd :org 1] inc)))))

(defn slide-left-field [field empty]
  (assoc field :body
         (map #(concat (next %) [empty])
              (:body field))))

(defn slide-left
 ([x] (slide-left x 0))
 ([{field :field cmd :cmd :as x} empty]
  (if (or (= (get-in cmd [:org 0]) 0)
          (and (= (:cmd cmd) :move-x)
               (= (:dst cmd) 0))
          (some (partial not= empty)
                (map first (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-x)
          (update-in x [:cmd :dst] dec)
          x)
        (update-in [:field] #(slide-left-field % empty))
        (update-in [:cmd :org 0] dec)))))

(defn slide-right-field [field empty]
  (assoc field :body
         (map #(cons empty (butlast %))
              (:body field))))

(defn slide-right
 ([x] (slide-right x 0))
 ([{field :field cmd :cmd :as x} empty]
  (if (or (<= (count (first (:body field))) (get-in cmd [:org 0]))
          (and (= (:cmd cmd) :move-x)
               (<= (count (first (:body field))) (:dst cmd)))
          (some (partial not= empty)
                (map last (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-x)
          (update-in x [:cmd :dst] inc)
          x)
        (update-in [:field] #(slide-right-field % empty))
        (update-in [:cmd :org 0] inc)))))

(defn expand-v
 ([x] (expand-v x 0)) ; x -> {field :field cmd :cmd}
 ([x empty]
  (concat (reverse (take-while identity (iterate #(slide-upper % empty) x)))
          (next (take-while identity (iterate #(slide-lower % empty) x))))))

(defn expand-h
 ([x] (expand-h x 0)) ; x -> {field :field cmd :cmd}
 ([x empty]
  (concat (reverse (take-while identity (iterate #(slide-left % empty) x)))
          (next (take-while identity (iterate #(slide-right % empty) x))))))

(defn expand
 ([x] (expand x 0))
 ([x empty] (mapcat #(expand-h % empty) (expand-v x empty))))

(defn one-hot [val len]
  (take len (concat (repeat val 0) [1] (repeat 0))))

(defn mlp-input-field [{body :body}]
  (mapcat {0 [0 0] 1 [0 1] 2 [1 0] 3 [1 1]}
          (apply concat body)))

(defn mlp-input-cmd [{cmd :cmd [x y] :org dst :dst} [cx cy]]
  (concat (case cmd :move-x [1 0] [0 1])
          (one-hot x cx)
          (one-hot y cy)
          (one-hot dst (max cx cy))))

(defn mlp-input [{field :field cmd :cmd}] 
  {:niv (mlp-input-field field)
   :eov (mlp-input-cmd cmd [(count (first (:body field)))
                            (count (:body field))])})
