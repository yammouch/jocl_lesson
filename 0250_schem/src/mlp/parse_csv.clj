(ns mlp.parse-csv
 (:require [clojure.pprint])
 (:require [mlp.parsec :as psc]))

(def space1 (psc/char (set (seq " \t\r"))))
(def comma (psc/char (partial = \,)))

(defn cell [s]
  (let [f (psc/all (psc/times space1 0 nil)
                   (psc/times (psc/char (comp not (set (seq "\n \t\r,"))))
                              0 nil)
                   (psc/times space1 0 nil))
        [parsed remain] (f s)]
    (if parsed
      [(apply str (nth parsed 1)) remain]
      [nil remain])))

(defn line [s]
  (let [f (psc/all cell
                   (psc/times (psc/all comma cell) 0 nil))
        [parsed remain] (f s)]
    (if parsed
      [(vec (cons (first parsed)
                  (as-> parsed p
                        (nth p 1)
                        (map #(nth % 1) p))))
       remain]
      [nil remain])))

(def csv (psc/times line 0 nil))

(defn csv [s]
  (let [f (psc/all line
                   (psc/times (psc/all (psc/char #{\newline}) line) 0 nil))
        [[p ps :as parsed] remain] (f s)]
    (if parsed
      [(vec (cons p (map #(nth % 1) ps)))
       remain]
      [nil remain])))
    
