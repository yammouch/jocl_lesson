(ns jocl-lesson.cl)

; very thin wrapper of OpenCL API

(defn clGetPlatformIDs []
  (let [num-entries 256
        platforms (make-array org.jocl.cl_platform_id num-entries)
        num-platforms (int-array 1)
        errcode-ret (org.jocl.CL/clGetPlatformIDs
                     num-entries platforms num-platforms)]
    (if (= errcode-ret org.jocl.CL/CL_SUCCESS)
      (take (nth num-platforms 0) platforms)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetPlatformInfo [platform param-name]
  (let [param-value-size 65536
        errcode-ret (int-array 1)
        param-value-body (byte-array param-value-size)
        param-value (org.jocl.Pointer/to param-value-body)
        param-value-size-ret (long-array 1)]
    (org.jocl.CL/clGetPlatformInfo
     platform    
     (.get (.getField org.jocl.CL (str param-name)) nil)
     param-value-size
     param-value
     param-value-size-ret)
    (if (= (nth errcode-ret 0) org.jocl.CL/CL_SUCCESS)
      (take (nth param-value-size-ret 0)
            param-value-body)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetDeviceIDs [platform]
  (let [num-devices (int-array 1)
        _ (org.jocl.CL/clGetDeviceIDs
           platform org.jocl.CL/CL_DEVICE_TYPE_ALL 0 nil num-devices)
        devices (make-array org.jocl.cl_device_id (nth num-devices 0))
        errcode-ret (org.jocl.CL/clGetDeviceIDs
                     platform
                     org.jocl.CL/CL_DEVICE_TYPE_ALL
                     (nth num-devices 0)
                     devices
                     num-devices)]
    (if (= errcode-ret org.jocl.CL/CL_SUCCESS)
      (seq devices)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetDeviceInfo [device param-name]
  (let [param-value-size 65536
        param-value-body (byte-array param-value-size)
        param-value (org.jocl.Pointer/to param-value-body)
        param-value-size-ret (long-array 1)
        errcode-ret (org.jocl.CL/clGetDeviceInfo
                     device
                     (.get (.getField org.jocl.CL (str param-name)) nil)
                     param-value-size
                     param-value
                     param-value-size-ret)]
    (if (= errcode-ret org.jocl.CL/CL_SUCCESS)
      (take (nth param-value-size-ret 0) param-value-body)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clCreateContext [devices]
  (let [errcode-ret (int-array 1)
        context
        (org.jocl.CL/clCreateContext
          nil             ; const cl_context_properties *properties
          (count devices) ; cl_uint num_devices
          (into-array org.jocl.cl_device_id devices)
          ; const cl_device_id *devices
          nil             ; (void CL_CALLBACK *pfn_notiry) (
                          ;   const char *errinfo,
                          ;   const void *private_info,
                          ;   size_t cb,
                          ;   void *user_data)
          nil             ; void *user_data
          errcode-ret     ; cl_int *errcode_ret
          )]
    (if (= (nth errcode-ret 0) org.jocl.CL/CL_SUCCESS)
      context
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clCreateCommandQueue [context device]
  (let [errcode-ret (int-array 1)
        queue (org.jocl.CL/clCreateCommandQueue
               context device
               0  ; const cl_queue_properties *properties
               errcode-ret)]
    (if (= (nth errcode-ret 0) org.jocl.CL/CL_SUCCESS)
      queue
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetProgramInfo [program param-name]
  (let [param-value-size 65536
        param-value-body (byte-array param-value-size)
        param-value (org.jocl.Pointer/to param-value-body)
        param-value-size-ret (long-array 1)
        errcode-ret (org.jocl.CL/clGetProgramInfo
                     program
                     (.get (.getField org.jocl.CL (str param-name)) nil)
                     param-value-size
                     param-value
                     param-value-size-ret)]
    (if (= errcode-ret org.jocl.CL/CL_SUCCESS)
      (take (nth param-value-size-ret 0) param-value-body)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetProgramBuildInfo [program device param-name]
  (let [param-value-size 65536
        param-value-body (byte-array param-value-size)
        param-value (org.jocl.Pointer/to param-value-body)
        param-value-size-ret (long-array 1)
        errcode-ret (org.jocl.CL/clGetProgramBuildInfo
                     program
                     device
                     (.get (.getField org.jocl.CL (str param-name)) nil)
                     param-value-size
                     param-value
                     param-value-size-ret)]
    (if (= errcode-ret org.jocl.CL/CL_SUCCESS)
      (take (nth param-value-size-ret 0) param-value-body)
      (throw (Exception. (org.jocl.CL/stringFor_errorCode errcode-ret)))
      )))

; subroutines for get bunch of OpenCL infomation

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

(defn parse-unsigned-info [array]
  (reduce (fn [acc x] (+ (* 256 acc) x))
          (reverse (map (fn [x] (if (neg? x) (+ x 256) x))
                        array))))
(defn parse-str-info [array]
  (apply str (map char (butlast array))))
(defn parse-device-type [array]
  (let [types (map #(symbol (str "CL_DEVICE_TYPE_" %))
                   '[DEFAULT CPU GPU ACCELERATOR])
        type-vals (map #(.get (.getField org.jocl.CL (str %)) nil)
                       types)
        u (parse-unsigned-info array)]
    (vec (map first
              (remove #(= 0 (get % 1))
                      (map (fn [t tv] [t (bit-and u tv)]) types type-vals)
                      )))))
(defn parse-size-t-array [array]
  (vec (map parse-unsigned-info
            (partition org.jocl.Sizeof/size_t array)
            )))

;(defn device-info [device]
;  (let [long-info (map #(clGetDeviceInfo device %)
;                       long-props)
;        str-info (map #(clGetDeviceInfo device %)
;                      str-props)
;        hex-info (map #(clGetDeviceInfo device %)
;                      hex-props)]
;    (concat (map vector long-props (map parse-unsigned-info long-info))
;            (map vector str-props (map parse-str-info str-info))
;            (map vector hex-props (map parse-unsigned-info hex-info))
;            [['CL_DEVICE_TYPE
;              (parse-device-type (clGetDeviceInfo device 'CL_DEVICE_TYPE))]
;             ['CL_DEVICE_MAX_WORK_ITEM_SIZES
;              (parse-size-t-array
;               (clGetDeviceInfo device 'CL_DEVICE_MAX_WORK_ITEM_SIZES))]])))

(defn get-device [device]
  (let [long-info (map #(clGetDeviceInfo device %)
                       long-props)
        str-info (map #(clGetDeviceInfo device %)
                      str-props)
        hex-info (map #(clGetDeviceInfo device %)
                      hex-props)]
    {:id   device
     :info (concat (map vector long-props (map parse-unsigned-info long-info))
                   (map vector str-props (map parse-str-info str-info))
                   (map vector hex-props (map parse-unsigned-info hex-info))
                   [['CL_DEVICE_TYPE
                    (parse-device-type
                     (clGetDeviceInfo device 'CL_DEVICE_TYPE))]
                    ['CL_DEVICE_MAX_WORK_ITEM_SIZES
                     (parse-size-t-array
                      (clGetDeviceInfo device
                       'CL_DEVICE_MAX_WORK_ITEM_SIZES))]])}))

(defn get-platform [platform]
  (let [names '[CL_PLATFORM_PROFILE
                CL_PLATFORM_VERSION
                CL_PLATFORM_NAME
                CL_PLATFORM_VENDOR
                CL_PLATFORM_EXTENSIONS]]
    {:id      platform
     :info    (concat 
               (map vector
                names
                (map #(parse-str-info (clGetPlatformInfo platform %))
                     names)))
     :devices (map get-device (clGetDeviceIDs platform))
     }))

(defn get-platforms [] (map get-platform (clGetPlatformIDs)))

(defn find-devices [type platform]
  (filter (fn [d] 
            (some #(= % type)
                  ((into {} (d :info)) 'CL_DEVICE_TYPE)
                  ))
          (platform :devices)))

(defn context [device-type]
  (loop [pfs (get-platforms)]
    (if (empty? pfs)
      nil
      (let [pf (first pfs)
            cpu (first (find-devices device-type pf))]
        (if cpu
          (let [context (clCreateContext [(cpu :id)])
                queue   (clCreateCommandQueue context (cpu :id))]
            {:platform pf
             :device   cpu
             :context  context
             :queue    queue})
          (recur (next pfs))
          )))))
