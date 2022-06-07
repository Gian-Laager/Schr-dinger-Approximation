package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	hBar                     = 1.0546e-34
	integrateSteps           = 10000
	maxGoRoutinesPerIntegral = 100
	maxGoRoutinesPerWaveFunc = 1000
	numPoints                = 100000
	x0                       = 0.0
)

func integrate(f func(float64) complex128, a float64, b float64, n int64) complex128 {
	if a > b {
		tmp := a
		a = b
		b = tmp
	}
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

func constructWaveFunc(mass float64, energy float64, c0 float64, theta float64, potential func(float64) float64) func(float64) complex128 {
	phase := func(x float64) complex128 {
		return cmplx.Sqrt(complex(potential(x)-energy, 0)) // / complex(hBar, 0)
	}

	cPlus := complex(0.5*c0*math.Cos(theta-math.Pi/4.0), 0)
	cMinus := complex(-0.5*c0*math.Sin(theta-math.Pi/4.0), 0)

	return func(x float64) complex128 {
		integral := integrate(phase, x0, x, integrateSteps)

		return (cPlus*cmplx.Exp(integral) + cMinus*cmplx.Exp(-integral)) / cmplx.Sqrt(integral)
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

func main() {
	mass := 0.5
	energy := 1e3
	c0 := 1.0
	theta := math.Pi / 5

	waveFunc := constructWaveFunc(mass, energy, c0, theta, squarePot)

	file, _ := os.Create("data.txt")

	psiChan := make(chan Pair[float64, complex128])

	const view = 0.9

	for i := int64(0); i < numPoints; i += numPoints / maxGoRoutinesPerWaveFunc {
		go func(start int64, end int64) {
			for k := start; k < end; k++ {
				x := float64(k)/float64(numPoints-1)*(view*2) - view
				psiChan <- Pair[float64, complex128]{x, waveFunc(x)}
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

	psis = append(psis[:len(psis)/2-1000], psis[len(psis)/2+1000:]...)

	for _, psi := range psis {
		x := psi.First
		psiOfX := psi.Second
		line := strings.Builder{}

		if !math.IsNaN(real(psiOfX)) && !math.IsNaN(imag(psiOfX)) {
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

	file.WriteString("\n\n")

	for _, psi := range psis {
		x := psi.First
		line := strings.Builder{}

		line.WriteString(strconv.FormatFloat(x, 'g', 17, 64))
		line.WriteString(" ")
		line.WriteString(strconv.FormatFloat(0.0, 'g', 17, 64))
		line.WriteString(" ")
		line.WriteString(strconv.FormatFloat(squarePot(x), 'g', 17, 64))
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
