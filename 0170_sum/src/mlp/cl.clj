(ns mlp.cl)

(import '(org.jocl CL Sizeof Pointer cl_device_id cl_event))

; utilities for this namespace

(defn handle-cl-error [err-code]
  (when (not= err-code CL/CL_SUCCESS)
    (throw (Exception. (CL/stringFor_errorCode err-code)))))

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

; thin wrappers

(defn query
 ([f]      (query f Byte/TYPE Long/TYPE   ))
 ([f tret] (query f tret      Integer/TYPE))
 ([f tret tsize]
  (let [size (make-array tsize 1)
        _ (handle-cl-error (f 0 nil size))
        body (make-array tret (first size))]
    (handle-cl-error (f (first size)
                        (if (= tret Byte/TYPE) (Pointer/to body) body)
                        nil))
    body)))

(defn clGetPlatformIDs []
  (query #(CL/clGetPlatformIDs %1 %2 %3) org.jocl.cl_platform_id))
(defn clCreateKernelsInProgram [program]
  (query #(CL/clCreateKernelsInProgram program %1 %2 %3) org.jocl.cl_kernel))

; very thin wrapper of OpenCL API

(defn clGetDeviceIDs [platform]
  (query #(CL/clGetDeviceIDs platform CL/CL_DEVICE_TYPE_ALL %1 %2 %3)
         cl_device_id))

(defn clCreateContext [devices]
  (let-err err
    [context (CL/clCreateContext nil (count devices)
              (into-array cl_device_id devices)
              nil nil err)]
    context))

(defn clCreateCommandQueue [context device]
  (let-err err
    [queue (CL/clCreateCommandQueue context device
            CL/CL_QUEUE_PROFILING_ENABLE err)]
    queue))

(defn clGetDeviceInfo [device param-name]
  (query #(CL/clGetDeviceInfo device param-name %1 %2 %3)))
(defn clGetPlatformInfo [platform param-name]
  (query #(CL/clGetPlatformInfo platform param-name %1 %2 %3)))
(defn clGetProgramInfo [program param-name]
  (query #(CL/clGetProgramInfo program param-name %1 %2 %3)))
(defn clGetProgramBuildInfo [program device param-name]
  (query #(CL/clGetProgramBuildInfo program device param-name %1 %2 %3)))
(defn clGetKernelInfo [kernel param-name]
  (query #(CL/clGetKernelInfo kernel param-name %1 %2 %3)))

; parser for queried info

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

; for program initialization

(defn find-devices [device-type platform]
  (let [ds (clGetDeviceIDs platform)
        pred (fn [d]
               (not= 0 (bit-and device-type
                        (parse-unsigned-info
                         (clGetDeviceInfo d CL/CL_DEVICE_TYPE)))))]
    (concat (filter pred ds)
            (filter (complement pred) ds)
            )))

(defn context [device-type]
  (loop [pfs (clGetPlatformIDs)]
    (if (empty? pfs)
      nil
      (let [pf (first pfs)
            dev (first (find-devices device-type pf))]
        (if dev
          (let [context (clCreateContext [dev])
                queue   (clCreateCommandQueue context dev)]
            {:platform pf
             :device   dev 
             :context  context
             :queue    queue})
          (recur (next pfs))
          )))))

; thick wrappers

(defn read-float [q mem n]
  (let [dbg-array (float-array n)]
    (handle-cl-error
     (CL/clEnqueueReadBuffer q mem CL/CL_TRUE
      0 (* (count dbg-array) Sizeof/cl_float) (Pointer/to dbg-array)
      0 nil nil))
    dbg-array))

(defn create-buffer [context type src]
  (let-err err
    [[unit-size ar-fn]
     (case type
       :f [Sizeof/cl_float float-array]
       (throw (Exception. "Illegal type in 'create-buffer'")))
     [ptr size flag]
     (if (coll? src)
       [(Pointer/to (ar-fn src)) (count src) CL/CL_MEM_COPY_HOST_PTR]
       [nil                      src         CL/CL_MEM_READ_WRITE   ])
     ret (CL/clCreateBuffer context flag (* unit-size size) ptr err)]
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
            nil nil nil)]
    (doseq [d devices]
      (println (parse-str-info
                (clGetProgramBuildInfo program d
                 CL/CL_PROGRAM_BUILD_LOG))))
    (handle-cl-error er)
    program))

(defn create-kernel [p name]
  (let-err err
    [ret (CL/clCreateKernel p name err)]
    ret))

(defn create-kernels-in-program [p]
  (into {}
        (map (fn [k] [(-> (clGetKernelInfo k CL/CL_KERNEL_FUNCTION_NAME)
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
