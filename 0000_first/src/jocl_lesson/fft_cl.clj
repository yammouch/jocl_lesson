(ns jocl-lesson.fft-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer cl_device_id cl_event))

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

(defn prepare-mem [context exp2]
  (let-err err
   [len (bit-shift-left 1 exp2)
    len-inv (float (/ 1.0 len))
    w-mem      (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
                (* len   Sizeof/cl_float) nil err)
    wave-mem   (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
                (* len   Sizeof/cl_float) nil err)
    buf0       (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
                (* len 2 Sizeof/cl_float) nil err)
    buf1       (CL/clCreateBuffer context CL/CL_MEM_READ_WRITE
                (* len 2 Sizeof/cl_float) nil err)
    result-mem (CL/clCreateBuffer context CL/CL_MEM_WRITE_ONLY
                (* len   Sizeof/cl_float) nil err)]
    {:w w-mem :wave wave-mem :buf0 buf0 :buf1 buf1 :result result-mem}))

(def kernel-source-code (slurp "fft.cl"))

(defn prepare-kernels [context devices]
  (let-err err
   [program (let [p (CL/clCreateProgramWithSource
                     context 1 (into-array String [kernel-source-code])
                     (long-array [(count kernel-source-code)]) err)
                  er (CL/clBuildProgram
                      p 1 (into-array cl_device_id devices)
                      nil ;(if simd "-D SIMD=1" nil)
                      nil nil)]
              (doseq [d devices]
                (println (cl/parse-str-info
                          (cl/clGetProgramBuildInfo p d
                           'CL_PROGRAM_BUILD_LOG))))
              p)
    make-w       (CL/clCreateKernel program "make_w" err) 
    step-1st     (CL/clCreateKernel program "step_1st" err) 
    step1        (CL/clCreateKernel program "step1" err) 
    post-process (CL/clCreateKernel program "post_process" err)]
     {:program program :make-w make-w :step-1st step-1st :step1 step1
      :post-process post-process}))

(defn call-make-w [q k w exp2 local-work-size events]
  (let [n-half (bit-shift-left 1 (dec exp2))
        local-work-size (min n-half 128)]
    (set-args k :m w :i exp2)
    (handle-cl-error
     (CL/clEnqueueNDRangeKernel q k 1
      nil (long-array [n-half]) (long-array [local-work-size])
      0 nil (first events)))
    (handle-cl-error (CL/clWaitForEvents 1 events))
    ))

(defn call-step-1st [q k src dst n-half events]
  (let [local-work-size (min n-half 128)]
    (set-args k :m src :m dst :i n-half)
    (handle-cl-error
     (CL/clEnqueueNDRangeKernel q k 1
      nil (long-array [n-half]) (long-array [local-work-size])
      0 nil (first events)))
    (handle-cl-error (CL/clWaitForEvents 1 events))
    ))

(defn call-step1 [q k src w dst n-half w-mask events]
  (let [local-work-size (min n-half 128)]
    (set-args k :m src :m w :m dst :i n-half :i w-mask)
    (handle-cl-error
     (CL/clEnqueueNDRangeKernel q k 1
      nil (long-array [n-half]) (long-array [local-work-size])
      0 nil (first events)))
    (handle-cl-error (CL/clWaitForEvents 1 events))
    ))

(defn call-post-process [q k src dst mag-0db-inv exp2 events]
  (let [n (bit-shift-left 1 exp2)
        local-work-size (min n 128)]
    (set-args k :m src :m dst :f mag-0db-inv :i exp2)
    (handle-cl-error
     (CL/clEnqueueNDRangeKernel q k 1
      nil (long-array [n]) (long-array [local-work-size])
      0 nil (first events)))
    (handle-cl-error (CL/clWaitForEvents 1 events))
    ))

(defn engine [ctx queue
              {make-w :make-w step-1st :step-1st step1 :step1
               post-process :post-process}
              {w :w wave :wave buf0 :buf0 buf1 :buf1 result :result}
              exp2 factor]
  (let [n      (bit-shift-left 1      exp2 )
        n-half (bit-shift-left 1 (dec exp2))
        err (int-array 1)
        event (CL/clCreateUserEvent ctx err)
        _ (handle-cl-error (nth err 0))
        events (into-array cl_event [event])
        local-work-size (min (bit-shift-left 1 exp2) 128)
        _ (call-make-w queue make-w w exp2 local-work-size events)
        _ (call-step-1st queue step-1st wave buf0 n-half events)
        butterflied
        (loop [i 1, src buf0, dst buf1, w-mask (int 1)]
          (if (<= exp2 i)
            src
            (do (call-step1 queue step1 src w dst n-half w-mask events)
                (recur (inc i) dst src (bit-or (bit-shift-left w-mask 1) 1))
                )))]
    (call-post-process queue post-process butterflied result factor exp2
                       events)))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (CL/clReleaseKernel (@cl-prg :make-w))
  (CL/clReleaseKernel (@cl-prg :step-1st))
  (CL/clReleaseKernel (@cl-prg :step1))
  (CL/clReleaseKernel (@cl-prg :post-process))
  (CL/clReleaseProgram (@cl-prg :program))
  (CL/clReleaseMemObject (@cl-mem :w))
  (CL/clReleaseMemObject (@cl-mem :wave))
  (CL/clReleaseMemObject (@cl-mem :buf0))
  (CL/clReleaseMemObject (@cl-mem :buf1))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(def exp2 (ref 12))

(defn init []
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context) @exp2))
    (ref-set cl-prg (prepare-kernels (@cl-env :context)
                                     [(get-in @cl-env [:device :id])]
                                     ))))

(defn fft-mag-norm [bytes ofs swing-0db]
  (let [n (bit-shift-left 1 @exp2)]
    (handle-cl-error
     (CL/clEnqueueWriteBuffer (:queue @cl-env) (:wave @cl-mem) CL/CL_TRUE
      0 (* n Sizeof/cl_float)
      (.withByteOffset (Pointer/to bytes) ofs)
      0 nil nil))
    (engine (:context @cl-env) (:queue @cl-env) @cl-prg @cl-mem @exp2
            (/ 2.0 swing-0db n))
    (read-float (:queue @cl-env) (:result @cl-mem) (bit-shift-left 1 @exp2))
    ))
