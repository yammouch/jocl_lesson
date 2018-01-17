(ns mlp.parse-csv
 (:require [clojure.pprint]))

(defn eos [s]
  (if (empty? s)
    [:eos s]
    [nil s]))

(defn ch1 [pred]
  (fn [s]
    (if (and (not (empty? s))
        (pred (first s)))
      [(first s) (rest s)]
      [nil s])))

(defn all [& ps]
  (fn [s]
    (loop [[p & p2 :as ps] ps, s s, acc []]
      (if (empty? ps)
        [acc s]
        (let [[parsed remain] (p s)]
          (if parsed
            (recur p2 remain (conj acc parsed))
            [nil remain]))))))

(defn oneof [& ps]
  (fn [s]
    (loop [[p & p2 :as ps] ps]
      (if (empty? ps)
        [nil s]
        (let [[parsed remain] (p s)]
          (if parsed
            [parsed remain]
            (recur p2)))))))

(defn times [p lo up]
  (fn [str]
    (loop [i 0, acc [], remain str]
      (if (<= lo i)
        (loop [i lo, acc acc, remain remain]
          (if (and up (<= up i))
            [acc remain]
            (let [[parsed remain] (p remain)]
              (if parsed
                (recur (+ i 1) (conj acc parsed) remain)
                [acc remain]))))
        (let [[parsed remain] (p remain)]
          (if parsed
            (recur (+ i 1) (conj acc parsed) remain)
            [nil remain])))))) 

(def space1 (ch1 (set (seq " \t\r"))))
(def comma (ch1 (partial = \,)))

(defn cell [s]
  (let [f (all (times space1 0 nil)
               (times (ch1 (comp not (set (seq "\n \t\r,"))))
                      0 nil)
               (times space1 0 nil))
        [parsed remain] (f s)]
    (if parsed
      [(apply str (nth parsed 1)) remain]
      [nil remain])))

(defn line [s]
  (let [f (all cell
               (times (all comma cell) 0 nil))
        [parsed remain] (f s)]
    (if parsed
      [(vec (cons (first parsed)
                  (as-> parsed p
                        (nth p 1)
                        (map #(nth % 1) p))))
       remain]
      [nil remain])))

(def csv (times line 0 nil))

(defn csv [s]
  (let [f (all line
               (times (all (ch1 #{\newline}) line) 0 nil))
        [[p ps :as parsed] remain] (f s)]
    (if parsed
      [(vec (cons p (map #(nth % 1) ps)))
       remain]
      [nil remain])))
 
