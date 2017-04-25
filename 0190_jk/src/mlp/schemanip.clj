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
  (if (or (<= (get-in field [:size 1]) (get-in cmd [:org 1]))
          (and (= (:cmd cmd) :move-y)
               (<= (get-in field [:size 1]) (:dst cmd)))
          (some (partial not= empty)
                (last (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-y)
          (update-in x [:cmd :dst] inc)
          x)
        (update-in [:field] #(slide-lower-field % empty))
        (update-in [:cmd :org 1] inc)))))

(defn slide-left-field [field]
  (assoc field :body
         (map #(concat (next %)
                       [(repeat (count (get-in field [:body 0 0]))
                                0)])
              (:body field))))

(defn slide-left [{field :field cmd :cmd :as x}]
  (if (or (= (get-in cmd [:org 0]) 0)
          (and (= (:cmd cmd) :move-x)
               (= (:dst cmd) 0))
          (some (partial some (complement zero?))
                (map first (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-x)
          (update-in x [:cmd :dst] dec)
          x)
        (update-in [:field] slide-left-field)
        (update-in [:cmd :org 0] dec))))

(defn slide-right-field [field]
  (assoc field :body
         (map #(cons (repeat (count (get-in field [:body 0 0])) 0)
                     (butlast %))
              (:body field))))

(defn slide-right [{field :field cmd :cmd :as x}]
  (if (or (<= (get-in field [:size 0]) (get-in cmd [:org 0]))
          (and (= (:cmd cmd) :move-x)
               (<= (get-in field [:size 0]) (:dst cmd)))
          (some (partial some (complement zero?))
                (map last (:body field))))
    nil
    (-> (if (= (:cmd cmd) :move-x)
          (update-in x [:cmd :dst] inc)
          x)
        (update-in [:field] slide-right-field)
        (update-in [:cmd :org 0] inc))))

(defn expand-v [x] ; x -> {field :field cmd :cmd}
  (concat (reverse (take-while identity (iterate slide-upper x)))
          (next (take-while identity (iterate slide-lower x)))))

(defn expand-h [x] ; x -> {field :field cmd :cmd}
  (concat (reverse (take-while identity (iterate slide-left x)))
          (next (take-while identity (iterate slide-right x)))))

(defn expand [x] (mapcat expand-h (expand-v x)))

(defn one-hot [val len]
  (take len (concat (repeat val 0) [1] (repeat 0))))

(defn mlp-input-field [{body :body}]
  (apply concat (apply concat body)))

(defn mlp-input-cmd [{cmd :cmd [x y] :org dst :dst} [cx cy]]
  (concat (case cmd :move-x [1 0] [0 1])
          (one-hot x cx)
          (one-hot y cy)
          (one-hot dst (max cx cy))))

(defn mlp-input [{field :field cmd :cmd}] 
  {:niv (mlp-input-field field)
   :eov (mlp-input-cmd cmd (:size field))})
