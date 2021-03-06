(ns mlp.cl)

(import '(org.jocl CL Sizeof Pointer cl_device_id cl_event))

; very thin wrapper of OpenCL API

(defn clGetViaPointer [f type]
  (let [num 1024
        ar (make-array type num)
        num-ret (int-array 1)
        errcode-ret (f num ar num-ret)]
    (if (= errcode-ret CL/CL_SUCCESS)
      (take (nth num-ret 0) ar)
      (throw (Exception. (CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetPlatformIDs []
  (clGetViaPointer #(CL/clGetPlatformIDs %1 %2 %3) org.jocl.cl_platform_id))
(defn clCreateKernelsInProgram [program]
  (clGetViaPointer #(CL/clCreateKernelsInProgram program %1 %2 %3)
                   org.jocl.cl_kernel))

(defn clGetDeviceIDs [platform]
  (let [num-devices (int-array 1)
        _ (CL/clGetDeviceIDs
           platform CL/CL_DEVICE_TYPE_ALL 0 nil num-devices)
        devices (make-array cl_device_id (nth num-devices 0))
        errcode-ret (CL/clGetDeviceIDs
                     platform
                     CL/CL_DEVICE_TYPE_ALL
                     (nth num-devices 0)
                     devices
                     num-devices)]
    (if (= errcode-ret CL/CL_SUCCESS)
      (seq devices)
      (throw (Exception. (CL/stringFor_errorCode errcode-ret)))
      )))

(defn clCreateContext [devices]
  (let [errcode-ret (int-array 1)
        context
        (CL/clCreateContext
          nil             ; const cl_context_properties *properties
          (count devices) ; cl_uint num_devices
          (into-array cl_device_id devices)
          ; const cl_device_id *devices
          nil             ; (void CL_CALLBACK *pfn_notiry) (
                          ;   const char *errinfo,
                          ;   const void *private_info,
                          ;   size_t cb,
                          ;   void *user_data)
          nil             ; void *user_data
          errcode-ret     ; cl_int *errcode_ret
          )]
    (if (= (nth errcode-ret 0) CL/CL_SUCCESS)
      context
      (throw (Exception. (CL/stringFor_errorCode errcode-ret)))
      )))

(defn clCreateCommandQueue [context device]
  (let [errcode-ret (int-array 1)
        queue (CL/clCreateCommandQueue
               context device
               CL/CL_QUEUE_PROFILING_ENABLE
               errcode-ret)]
    (if (= (nth errcode-ret 0) CL/CL_SUCCESS)
      queue
      (throw (Exception. (CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetAnInfo [f param-name]
  (let [param-value-size 65536
        param-value-body (byte-array param-value-size)
        param-value (Pointer/to param-value-body)
        param-value-size-ret (long-array 1)
        errcode-ret (f
                     (.get (.getField CL (str param-name)) nil)
                     param-value-size
                     param-value
                     param-value-size-ret)]
    (if (= errcode-ret CL/CL_SUCCESS)
      (take (nth param-value-size-ret 0) param-value-body)
      (throw (Exception. (CL/stringFor_errorCode errcode-ret)))
      )))

(defn clGetDeviceInfo [device param-name]
  (clGetAnInfo #(CL/clGetDeviceInfo device %1 %2 %3 %4) param-name))
(defn clGetPlatformInfo [platform param-name]
  (clGetAnInfo #(CL/clGetPlatformInfo platform %1 %2 %3 %4) param-name))
(defn clGetProgramInfo [program param-name]
  (clGetAnInfo #(CL/clGetProgramInfo program %1 %2 %3 %4) param-name))
(defn clGetProgramBuildInfo [program device param-name]
  (clGetAnInfo #(CL/clGetProgramBuildInfo program device %1 %2 %3 %4)
               param-name))
(defn clGetKernelInfo [kernel param-name]
  (clGetAnInfo #(CL/clGetKernelInfo kernel %1 %2 %3 %4) param-name))

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
        type-vals (map #(.get (.getField CL (str %)) nil)
                       types)
        u (parse-unsigned-info array)]
    (vec (map first
              (remove #(= 0 (get % 1))
                      (map (fn [t tv] [t (bit-and u tv)]) types type-vals)
                      )))))
(defn parse-size-t-array [array]
  (vec (map parse-unsigned-info
            (partition Sizeof/size_t array)
            )))

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
  (let [ds (platform :devices)
        pred (fn [d]
               (some #(= % type)
                     ((into {} (d :info)) 'CL_DEVICE_TYPE)
                     ))]
    (concat (filter pred ds)
            (filter (complement pred) ds)
            )))

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

(defn handle-cl-error [err-code]
  (when (not= err-code CL/CL_SUCCESS)
    (throw (Exception. (CL/stringFor_errorCode err-code)))))

(defn read-float [q mem n]
  (let [dbg-array (float-array n)]
    (handle-cl-error
     (CL/clEnqueueReadBuffer q mem CL/CL_TRUE
      0 (* (count dbg-array) Sizeof/cl_float) (Pointer/to dbg-array)
      0 nil nil))
    dbg-array))

(defn find-symbol [s x]
  (if (coll? x)
    (or (find-symbol s (first x))
        (find-symbol s (next  x)))
    (= x s)))

(defmacro let-err [err-name binds & body]
  `(let [~err-name (int-array 1)
         ~@(apply concat
            (map (fn [[var clause]]
                   (if (find-symbol err-name clause)
                     `(~var (let [ret# ~clause]
                              (handle-cl-error (first ~err-name))
                              ret#))
                     `(~var ~clause)
                     ))
                 (partition 2 binds)))]
     ~@body))

(defn create-buffer [context type src]
  (let [err (int-array 1)
        [unit-size ar-fn]
        (case type
          :f [Sizeof/cl_float float-array]
          (throw (Exception. "Illegal type in 'create-buffer'")))
        [ptr size flag]
        (if (coll? src)
          [(Pointer/to (ar-fn src)) (count src) CL/CL_MEM_COPY_HOST_PTR]
          [nil                      src         CL/CL_MEM_READ_WRITE   ])
        ret (CL/clCreateBuffer context flag (* unit-size size) ptr err)]
    (handle-cl-error (first err))
    ret))

(defn set-args [kernel & args]
  (doseq [[i type arg] (map cons (range) (partition 2 args))]
    (let [[size pt-src] (case type
                          :f [Sizeof/cl_float (float-array [arg])]
                          :i [Sizeof/cl_int   (int-array   [arg])]
                          :m [Sizeof/cl_mem                 arg  ]
                          (throw (Exception. "Illegal type in 'set-args'"))
                          )]
      (handle-cl-error
        (CL/clSetKernelArg kernel i size (Pointer/to pt-src))
        ))))

(defn compile-kernel-source [context devices source]
  (let [err (int-array 1)
        program (CL/clCreateProgramWithSource
                 context 1 (into-array String [source])
                 (long-array [(count source)]) err)
        er (CL/clBuildProgram
            program 1 (into-array cl_device_id devices)
            nil ;(if simd "-D SIMD=1" nil)
            nil nil)]
    (doseq [d devices]
      (println (parse-str-info
                (clGetProgramBuildInfo program d
                 'CL_PROGRAM_BUILD_LOG))))
    (handle-cl-error er)
    program))

(defn create-kernel [p name]
  (let [err (int-array 1)
        ret (CL/clCreateKernel p name err)]
    (handle-cl-error (first err))
    ret))

(defn create-kernels-in-program [p]
  (into {}
        (map (fn [k] [(-> (clGetKernelInfo k 'CL_KERNEL_FUNCTION_NAME)
                          parse-str-info)
                      k])
             (clCreateKernelsInProgram p))))

(defn callk [q k global-work-offset global-work-size & args]
  (apply set-args k args)
  (handle-cl-error
   (CL/clEnqueueNDRangeKernel q k (count global-work-size)
    (if global-work-offset (long-array global-work-offset) nil)
    (long-array global-work-size) nil
    0 nil nil)))
