;;; test test
;;;
;;;Code:

(define (complex-plot t0 tmax step f)
  (unless (> t0 tmax)
    (let ((z (f t0)))
      (display t0)
      (display " ")
      (display (real-part z))
      (display " ")
      (display (imag-part z))
      (display "\r\n")
      (complex-plot (+ t0 step) tmax step f))))
(with-output-to-file "out.txt" (lambda() (complex-plot 0 25 (/ 3.1415 32) lambda (t) (exp(* 0+i t)))))
