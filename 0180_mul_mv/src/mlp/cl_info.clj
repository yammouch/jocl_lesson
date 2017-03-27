(ns mlp.cl-info
  (:gen-class)
  (:import  [org.jocl CL])
  (:require [mlp.cl :as cl]
            [clojure.pprint]))

(def long-props (map #(symbol (str "CL_DEVICE_" %))
                '[VENDOR_ID
                  MAX_COMPUTE_UNITS
                  MAX_WORK_ITEM_DIMENSIONS
                  MAX_WORK_GROUP_SIZE
                  PREFERRED_VECTOR_WIDTH_CHAR
                  PREFERRED_VECTOR_WIDTH_SHORT
                  PREFERRED_VECTOR_WIDTH_INT
                  PREFERRED_VECTOR_WIDTH_FLOAT
                  PREFERRED_VECTOR_WIDTH_DOUBLE
                  MAX_CLOCK_FREQUENCY
                  ADDRESS_BITS
                  MAX_MEM_ALLOC_SIZE
                  IMAGE_SUPPORT
                  MAX_READ_IMAGE_ARGS
                  MAX_WRITE_IMAGE_ARGS
                  IMAGE2D_MAX_WIDTH
                  IMAGE2D_MAX_HEIGHT
                  IMAGE3D_MAX_WIDTH
                  IMAGE3D_MAX_HEIGHT
                  IMAGE3D_MAX_DEPTH
                  MAX_SAMPLERS
                  MAX_PARAMETER_SIZE
                  MEM_BASE_ADDR_ALIGN
                  MIN_DATA_TYPE_ALIGN_SIZE
                  GLOBAL_MEM_CACHELINE_SIZE
                  GLOBAL_MEM_CACHE_SIZE
                  GLOBAL_MEM_SIZE
                  MAX_CONSTANT_BUFFER_SIZE
                  MAX_CONSTANT_ARGS
                  LOCAL_MEM_SIZE
                  ERROR_CORRECTION_SUPPORT
                  PROFILING_TIMER_RESOLUTION
                  ENDIAN_LITTLE
                  AVAILABLE
                  COMPILER_AVAILABLE]))

(def str-props (map #(symbol (str "CL_DEVICE_" %))
               '[NAME
                 VENDOR
                 PROFILE
                 VERSION
                 EXTENSIONS]))

(def hex-props (map #(symbol (str "CL_DEVICE_" %))
               '[SINGLE_FP_CONFIG
                 QUEUE_PROPERTIES]))

(defn print-device [device]
  (let [long-info (map #(cl/clGetDeviceInfo device
                         (.get (.getField CL (str %)) nil))
                       long-props)
        str-info (map #(cl/clGetDeviceInfo device
                        (.get (.getField CL (str %)) nil))
                      str-props)
        hex-info (map #(cl/clGetDeviceInfo device
                        (.get (.getField CL (str %)) nil))
                      hex-props)]
    (clojure.pprint/pprint device)
    (clojure.pprint/pprint
     (concat (map vector long-props (map cl/parse-unsigned-info long-info))
             (map vector str-props (map cl/parse-str-info str-info))
             (map vector hex-props (map cl/parse-unsigned-info hex-info))
             [['CL_DEVICE_TYPE
              (cl/parse-device-type
               (cl/clGetDeviceInfo device CL/CL_DEVICE_TYPE))]
              ['CL_DEVICE_MAX_WORK_ITEM_SIZES
               (cl/parse-size-t-array
                (cl/clGetDeviceInfo device
                 CL/CL_DEVICE_MAX_WORK_ITEM_SIZES))]]))))

(defn print-platform [platform]
  (let [names '[CL_PLATFORM_PROFILE
                CL_PLATFORM_VERSION
                CL_PLATFORM_NAME
                CL_PLATFORM_VENDOR
                CL_PLATFORM_EXTENSIONS]]
    (clojure.pprint/pprint platform)
    (clojure.pprint/pprint
     (concat 
      (map vector
       names
       (map #(cl/parse-str-info
              (cl/clGetPlatformInfo platform
               (.get (.getField CL (str %)) nil)))
            names))))
    (doseq [d (cl/clGetDeviceIDs platform)]
      (print-device d))))

(defn -main [& _]
  (doseq [p (cl/clGetPlatformIDs)]
    (print-platform p)))
