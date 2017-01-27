(ns jocl-lesson.mlp-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context]
  (into {}
        (map (fn [k size]
               [k (cl/create-buffer context (* size Sizeof/cl_float))])
             []
             [])))

(def kernel-source-code (slurp "kernel.cl"))

(defn prepare-kernels [context devices]
  (let [program (cl/compile-kernel-source context devices kernel-source-code)]
    {:program program
     :kernels (into {}
                    (map (fn [[k name]] [k (cl/create-kernel program name)])
                         [[:set0             "set0"            ]
                          [:dense-fw         "dense_fw"        ]
                          [:dense-bw-m       "dense_bw_m"      ]
                          [:dense-bw-ofs     "dense_bw_ofs"    ]
                          [:dense-bw-v       "dense_bw_v"      ]
                          [:sigmoid-fw       "sigmoid_fw"      ]
                          [:sigmoid-bw       "sigmoid_bw"      ]
                          [:softmax-fw-step1 "softmax_fw_step1"]
                          [:softmax-fw-step2 "softmax_fw_step2"]
                          [:softmax-fw-step3 "softmax_fw_step3"]
                          [:quadratic-bw     "quadratic_bw"    ]
                          [:cross-entropy-bw "cross_entropy_bw"]
                          ]))}))

(def cl-env (ref nil))
(def cl-mem (ref nil))
(def cl-prg (ref nil))
(def cl-ker (ref nil))

(defn finalize []
  (CL/clFlush (@cl-env :queue))
  (CL/clFinish (@cl-env :queue))
  (doseq [[_ v] @cl-ker] (CL/clReleaseKernel v))
  (CL/clReleaseProgram @cl-prg)
  (doseq [[_ v] @cl-mem] (CL/clReleaseMemObject v))
  (CL/clReleaseCommandQueue (@cl-env :queue))
  (CL/clReleaseContext (@cl-env :context)))

(defn init []
  (dosync
    (ref-set cl-env (cl/context 'CL_DEVICE_TYPE_GPU))
    (ref-set cl-mem (prepare-mem (@cl-env :context)))
    (let [{p :program k :kernels}
          (prepare-kernels (@cl-env :context)
                           [(get-in @cl-env [:device :id])])]
      (ref-set cl-prg p)
      (ref-set cl-ker k))))
