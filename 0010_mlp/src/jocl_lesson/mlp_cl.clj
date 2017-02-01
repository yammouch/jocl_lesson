(ns jocl-lesson.mlp-cl
  (:gen-class))

(require 'jocl-lesson.cl)
(alias 'cl 'jocl-lesson.cl)

(import '(org.jocl CL Sizeof Pointer))

(defn prepare-mem [context]
  (into {}
        (map (fn [[k size]]
               [k (cl/create-buffer context :f size)])
             [[:i0  [0.0 ]]
              [:l0  [0.0 ]]
              [:i1  [0.25]]
              [:l1  [0.0 ]]
              [:i2  [0.75]]
              [:l2  [1.0 ]]
              [:i3  [1.0 ]]
              [:l3  [1.0 ]]
              [:w   [1.0 ]]
              [:b   [0.0 ]]
              [:z    1    ]
              [:a    1    ]
              [:v    1    ]
              [:wacc 1    ]
              [:bacc 1    ]
              ])))

(def kernel-source-code (slurp "kernel.cl"))

(defn prepare-kernels [context devices]
  (let [program (cl/compile-kernel-source context devices kernel-source-code)]
    {:program program
     :kernels (into {}
                    (map (fn [[k name]] [k (cl/create-kernel program name)])
                         [[:set0             "set0"            ]
                          [:add              "add"             ]
                          [:sub              "sub"             ]
                          [:dense-fw         "dense_fw"        ]
                          [:dense-bw-m       "dense_bw_m"      ]
                          [:dense-bw-m-ov    "dense_bw_m_ov"   ]
                          [:dense-bw-ofs     "dense_bw_ofs"    ]
                          [:dense-bw-ofs-ov  "dense_bw_ofs_ov" ]
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

(defn fw [in]
  (let [{q :queue} @cl-env
        {dense-fw :dense-fw sigmoid-fw :sigmoid-fw} @cl-ker
        {w :w b :b z :z a :a} @cl-mem]
    (cl/callk q dense-fw   nil [1] :m z :m in :m b :m w :i 1 :i 1)
    (cl/callk q sigmoid-fw nil [1] :m a :m z)
    ))

(defn bw
 ([in label] (bw in label false))
 ([in label is-1st?]
  (let [{q :queue} @cl-env
        {add              :add
         sub              :sub
         cross-entropy-bw :cross-entropy-bw
         dense-bw-m       :dense-bw-m
         dense-bw-m-ov    :dense-bw-m-ov} @cl-ker
        {a :a v :v w :w b :b wacc :wacc bacc :bacc} @cl-mem]
    (cl/callk q cross-entropy-bw nil [1] :m v :m a :m label :f 0.1)
    (if is-1st?
      (do (cl/callk q dense-bw-m-ov nil [1] :m wacc :m in :m v :i 1)
          (CL/clEnqueueCopyBuffer q v bacc 0 0 Sizeof/cl_float 0 nil nil))
      (do (cl/callk q dense-bw-m    nil [1] :m wacc :m in :m v :i 1)
          (cl/callk q add           nil [1] :m bacc :m v)
          )))))
