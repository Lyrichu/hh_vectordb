package test

import (
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"testing"
)

type Vector = basic.Vector

func TestVector(t *testing.T) {
	vec1 := Vector{
		ID:     0,
		Values: []float64{1.0, 2.0},
	}
	vec2 := Vector{
		ID:     1,
		Values: []float64{3.0, 4.0},
	}
	vec3 := Vector{
		ID:     2,
		Values: []float64{1.01, 2.89},
	}
	vec4 := Vector{
		ID:     3,
		Values: []float64{1.010, 2.8900},
	}
	assert.False(t, vec1.Equals(vec2))
	assert.True(t, vec3.Equals(vec4))
}
