package test

import (
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"math"
	"testing"
)

func TestEuclidDistance(t *testing.T) {
	arr1 := []float64{0.1, 0.2}
	arr2 := []float64{0.2, 1.0}
	res1 := math.Sqrt(0.1*0.1 + 0.8*0.8)
	assert.LessOrEqual(t, math.Abs(basic.EuclidDistance(arr1, arr2)-res1), 1e-6)
}
