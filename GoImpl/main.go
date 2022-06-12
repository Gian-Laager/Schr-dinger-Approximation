package main

import (
	"fmt"
	"gonum.org/v1/gonum/mathext"
	"math"
	"math/cmplx"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	hBar                     = 1
	integrateSteps           = 10000
	maxGoRoutinesPerIntegral = 100
	maxGoRoutinesPerWaveFunc = 100
	numPoints                = 10000
	x0                       = -10.0
	deltaDerivative          = 1e-9
	energy                   = 2 * math.Pi * hBar * (3 + 0.5)
	mass                     = 3.0
	c0                       = 1e-30
	theta                    = 1
	view                     = 5
	offset                   = -3
)

var (
	t1        = -math.Sqrt(energy + hBar*hBar/mass - math.Sqrt(hBar*hBar*(hBar*hBar+2*mass*energy))/mass)
	t2        = -math.Sqrt(energy + hBar*hBar/mass + math.Sqrt(hBar*hBar*(hBar*hBar+2*mass*energy))/mass)
	cPlus     = complex(0.5*c0*math.Cos(theta-math.Pi/4.0), 0)
	cMinus    = complex(-0.5*c0*math.Sin(theta-math.Pi/4.0), 0)
	potential = squarePot
)

func f(x float64) complex128 {
	return mathext.AiryAi(cmplx.Pow(complex(2*mass/(hBar*hBar)*(potential(t1)-potential(t2))/(t1-t2), 0), 1/3.0) * complex(x-(t1+t2)/2.0, 0))
}

func g(x float64) complex128 {
	return AiryBi(cmplx.Pow(complex(2*mass/(hBar*hBar)*(potential(t1)-potential(t2))/(t1-t2), 0), 1/3.0) * complex(x-(t1+t2)/2.0, 0))
}

func h(x float64) complex128 {
	integrant := func(x float64) complex128 {
		return cmplx.Sqrt(complex(2*mass*(potential(x)-energy), 0)) / hBar
	}

	return cmplx.Exp(Integral(integrant, x0, x, 1000)) / cmplx.Sqrt(integrant(x))
}

func k(x float64) complex128 {
	integrant := func(x float64) complex128 {
		return -cmplx.Sqrt(complex(2*mass*(potential(x)-energy), 0)) / hBar
	}

	return cmplx.Exp(Integral(integrant, x0, x, 1000)) / cmplx.Sqrt(integrant(x))
}

func ComputeCs() (complex128, complex128) {
	f1 := f(t1)
	g1 := g(t1)
	h1 := h(t1)
	k1 := k(t1)
	f2 := f(t2)
	g2 := g(t2)
	h2 := h(t2)
	k2 := k(t2)

	cA := ((-cMinus * (g1*k2 - g2*k1)) / (f1*g2 - f2*g1)) - ((cPlus * (g1*h2 - g2*h1)) / (f1*g2 - f2*g1))
	cB := ((cMinus * (f1*k2 - f2*k1)) / (f1*g2 - f2*g1)) + ((cPlus * (f1*h2 - f2*h1)) / (f1*g2 - f2*g1))
	return cA, cB
}

func AiryBi(z complex128) complex128 {
	return -complex(0, 1)*mathext.AiryAi(z) + 2*mathext.AiryAi(z*complex(-0.5, math.Sqrt(3)/2))*complex(math.Sqrt(3)/2.0, 0.5)
}

type Integrate struct {
	aPrev   float64
	bPrev   float64
	valPrev complex128
	n       int64

	f func(float64) complex128
}

func (c *Integrate) integrateAndUpdate(a float64, b float64, valPerv complex128) complex128 {
	c.aPrev = a
	c.bPrev = b
	c.valPrev = valPerv
	return valPerv
}

func (c *Integrate) Eval(a float64, b float64) complex128 {
	if a == c.aPrev && b == c.bPrev {
		return c.valPrev
	}

	if a < c.aPrev && b == c.bPrev {
		return c.integrateAndUpdate(a, b, c.valPrev+Integral(c.f, a, c.aPrev, c.n))
	}

	if a == c.aPrev && b > c.bPrev {
		return c.integrateAndUpdate(a, b, c.valPrev+Integral(c.f, c.bPrev, b, c.n))
	}

	if a < c.aPrev && b > c.bPrev {
		return c.integrateAndUpdate(a, b, Integral(c.f, a, c.aPrev, c.n)+c.valPrev+Integral(c.f, c.bPrev, b, c.n))
	}

	return c.integrateAndUpdate(a, b, Integral(c.f, a, b, c.n))
}

func Integral(f func(float64) complex128, a float64, b float64, n int64) complex128 {
	var result complex128 = 0.0
	var results chan complex128 = make(chan complex128)
	for i := int64(0); i < n; i += n / maxGoRoutinesPerIntegral {
		go func(start int64, stop int64) {
			partialResult := complex(0.0, 0.0)
			for k := start; k < stop; k++ {

				a_ := a + float64(k)*(b-a)/float64(n)
				b_ := a + (float64(k)+1)*(b-a)/float64(n)

				partialResult += complex(b_-a_, 0.0) * (f(a_) + f(b_)) / complex(2.0, 0.0)
			}
			results <- partialResult
		}(i, i+n/maxGoRoutinesPerIntegral)
	}
	for k := int64(0); k < maxGoRoutinesPerIntegral; k++ {
		result += <-results
	}
	return result
}

func isWkbValid(x float64, potential func(float64) float64, energy float64, mass float64) bool {
	//derivative := hBar / math.Sqrt(2*mass) * math.Abs(((potential(x+deltaDerivative)-energy)-(potential(x)-energy))/deltaDerivative)
	//qSquared := (potential(x) - energy) * (potential(x) - energy)
	//return derivative < qSquared
	var low float64
	var high float64

	if t1 > t2 {
		low = t2
		high = t1
	} else {
		low = t1
		high = t2
	}

	return x < low || x > high
}

func constructWaveFunc(mass float64, energy float64, c0 float64, theta float64, potential func(float64) float64) func(float64, *Integrate) complex128 {
	phase := func(x float64) complex128 {
		return cmplx.Sqrt(complex(2.0, 0)*complex(mass, 0)*complex(potential(x)-energy, 0)) / complex(hBar, 0)
	}

	//cPlus := complex(0.5*c0*math.Cos(theta-math.Pi/4.0), 0)
	//cMinus := complex(-0.5*c0*math.Sin(theta-math.Pi/4.0), 0)

	return func(x float64, integ *Integrate) complex128 {
		integ.f = phase
		integral := integ.Eval(x0, x)

		return (cPlus*cmplx.Exp(integral) + cMinus*cmplx.Exp(-integral)) / cmplx.Sqrt(phase(x))
	}
}

func stepPotential(x float64) float64 {
	if x < -1 {
		return 0
	} else if x < -0.5 {
		return 1
	} else if x < 0.0 {
		return 0.0
	} else if x < 0.5 {
		return 1
	} else {
		return 0
	}
}
func fmap(x float64, in_min float64, in_max float64, out_min float64, out_max float64) float64 {
	return (x-in_min)*(out_max-out_min)/(in_max-in_min) + out_min
}

type Pair[T, U any] struct {
	First  T
	Second U
}

func squarePot(x float64) float64 {
	return x * x
}

func constPot(x float64) float64 {
	return 0
}

func main() {
	println("t1 = ", t1)
	println("t2 = ", t2)
	println("x1 = ", (t1+t2)/2.0)

	waveFunc := constructWaveFunc(mass, energy, c0, theta, potential)

	file, _ := os.Create("data.txt")

	psiChan := make(chan Pair[float64, complex128])

	for i := int64(0); i < numPoints; i += numPoints / maxGoRoutinesPerWaveFunc {
		go func(start int64, end int64) {
			for k := start; k < end; k++ {

				integ := Integrate{aPrev: 0, bPrev: 0, valPrev: 0, n: integrateSteps}
				x := float64(k)/float64(numPoints-1)*(view*2) + offset - view
				psiChan <- Pair[float64, complex128]{x, waveFunc(x, &integ)}
			}
		}(i, i+numPoints/maxGoRoutinesPerWaveFunc)
	}
	psis := make([]Pair[float64, complex128], numPoints)
	for i := 0; i < numPoints; i++ {
		psis[i] = <-psiChan
	}

	println("Sorting")
	sort.Slice(psis, func(i, j int) bool {
		return psis[i].First < psis[j].First
	})

	for _, psi := range psis {
		x := psi.First
		psiOfX := psi.Second
		line := strings.Builder{}

		if !math.IsNaN(real(psiOfX)) && !math.IsNaN(imag(psiOfX)) && isWkbValid(x, potential, energy, mass) {
			line.WriteString(strconv.FormatFloat(x, 'g', 17, 64))
			line.WriteString(" ")
			line.WriteString(strconv.FormatFloat(real(psiOfX), 'g', 17, 64))
			line.WriteString(" ")
			line.WriteString(strconv.FormatFloat(imag(psiOfX), 'g', 17, 64))
			line.WriteString("\n")
		}
		_, err := file.WriteString(line.String())
		if err != nil {
			fmt.Errorf(err.Error())
			return
		}
	}
	//
	//file.WriteString("\n\n")
	//
	//const del = 1e-9
	//
	//for _, psi := range psis {
	//	x := psi.First
	//	line := strings.Builder{}
	//
	//	line.WriteString(strconv.FormatFloat(x, 'g', 17, 64))
	//	line.WriteString(" ")
	//	line.WriteString(strconv.FormatFloat(0.0, 'g', 17, 64))
	//	line.WriteString(" ")
	//	line.WriteString(strconv.FormatFloat(1/(2*mass)*math.Abs((potential(x+del)-energy)-(potential(x)-energy))/del, 'g', 17, 64))
	//	line.WriteString("\n")
	//	_, err := file.WriteString(line.String())
	//	if err != nil {
	//		fmt.Errorf(err.Error())
	//		return
	//	}
	//}

	file.WriteString("\n\n")

	cA, cB := ComputeCs()
	airyCorrection := func(x float64) complex128 {
		return cA*f(x) + cB*g(x)
	}

	for i := 0; i < 1000; i++ {
		x := fmap(float64(i), 0, 1000, t1, t2)
		y := airyCorrection(x)
		line := strings.Builder{}

		line.WriteString(strconv.FormatFloat(x, 'g', 17, 64))
		line.WriteString(" ")
		line.WriteString(strconv.FormatFloat(real(y), 'g', 17, 64))
		line.WriteString(" ")
		line.WriteString(strconv.FormatFloat(imag(y), 'g', 17, 64))
		line.WriteString("\n")
		_, err := file.WriteString(line.String())
		if err != nil {
			fmt.Errorf(err.Error())
			return
		}
	}

	err := file.Close()
	if err != nil {
		fmt.Errorf(err.Error())
		return
	}
}
