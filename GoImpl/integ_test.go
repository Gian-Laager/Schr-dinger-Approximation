package main

import (
	"math/cmplx"
	"testing"
)

func TestPerformance(t *testing.T) {
	main()
}

func testFunc(x float64) complex128 {
	return -cmplx.Exp(complex(-0.01*x*x, -x)) * complex(x, 50) / 5.0
}

func TestSquare(t *testing.T) {
	actual := func(a, b float64) complex128 {
		return 10*cmplx.Exp(complex(-0.01*b*b, -b)) - 10*cmplx.Exp(complex(-0.01*a*a, -a))
	}

	integ := Integrate{f: testFunc, valPrev: 0, aPrev: 0, bPrev: 0, n: 1000}

	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			result := integ.Eval(float64(j)/123.0, float64(i)/123.0)
			if cmplx.Abs(actual(float64(j)/123.0, float64(i)/123.0)-result) > 0.001 {
				t.Error("failed, expected: ", actual(0, float64(i)), ", result was: ", result)
			}
		}
	}
}
