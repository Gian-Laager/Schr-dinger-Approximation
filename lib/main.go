package main

import "C"
import "gonum.org/v1/gonum/mathext"

//export airy_ai
func airy_ai(zr float64, zi float64) (float64, float64) {
    z := mathext.AiryAi(complex(zr, zi))
    return real(z), imag(z)
}

func main() {

}