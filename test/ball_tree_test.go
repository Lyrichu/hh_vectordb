package test

import (
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"hh_vectordb/core"
	"math/rand"
	"testing"
	"time"
)

type BallTree = core.BallTree

func TestNewBallTree(t *testing.T) {
	vecs := []Vector{
		{
			0,
			[]float64{2, 3},
		},
		{
			1,
			[]float64{5, 4},
		},
		{
			2,
			[]float64{9, 6},
		},
		{
			3,
			[]float64{4, 7},
		},
		{
			4,
			[]float64{8, 1},
		},
		{
			5,
			[]float64{7, 2},
		},
	}
	ballTree := core.NewBallTree(vecs)
	assert.NotNil(t, ballTree)
}

func TestBallTreeInsert(t *testing.T) {
	ballTree := core.NewBallTree([]Vector{})
	vec := basic.GenerateRandomVector(0, 4, 1.0, 5.0)
	err := ballTree.Insert(vec)
	assert.Nil(t, err)
}

func TestBallTreeNearestV1(t *testing.T) {
	// 使用小规模固定向量来测试
	vecs := []Vector{
		{
			0,
			[]float64{2, 3},
		},
		{
			1,
			[]float64{5, 4},
		},
		{
			2,
			[]float64{9, 6},
		},
		{
			3,
			[]float64{4, 7},
		},
		{
			4,
			[]float64{8, 1},
		},
		{
			5,
			[]float64{7, 2},
		},
	}
	ballTree := core.NewBallTree(vecs)
	queryVec := Vector{6, []float64{8.1, 1.1}}
	resVec, err := ballTree.Nearest(queryVec)
	assert.Nil(t, err)
	assert.Equal(t, resVec, vecs[4])
}

func TestBallTreeNearestV2(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	numVectors := 10000
	vecDim := 10
	minValue := 1.0
	maxValue := 5.0

	tree := core.NewBallTree(nil)
	vectors := make([]Vector, numVectors)

	// 插入随机向量到 KDTree
	for i := 0; i < numVectors; i++ {
		vectors[i] = basic.GenerateRandomVector(int64(i), vecDim, minValue, maxValue)
		err := tree.Insert(vectors[i])
		assert.Nil(t, err)
	}

	for _, vec := range vectors {
		nearestVec, err := tree.Nearest(vec)
		assert.Nil(t, err)
		assert.True(t, vec.Equals(nearestVec))
		assert.Equal(t, vec.ID, nearestVec.ID)
	}
}

func TestBallTreeDelete(t *testing.T) {
	vecs := []Vector{
		{
			0,
			[]float64{2, 3},
		},
		{
			1,
			[]float64{5, 4},
		},
		{
			2,
			[]float64{9, 6},
		},
		{
			3,
			[]float64{4, 7},
		},
		{
			4,
			[]float64{8, 1},
		},
		{
			5,
			[]float64{7, 2},
		},
	}
	ballTree := core.NewBallTree(vecs)
	vecs1, err := ballTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 6)
	err = ballTree.Delete(Vector{2, []float64{9, 6}})
	assert.Nil(t, err)
	vecs1, err = ballTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)

	err = ballTree.Delete(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = ballTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 4)

	err = ballTree.Insert(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = ballTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)
}

func TestBallTreeVectors(t *testing.T) {
	vecs := []Vector{
		{
			0,
			[]float64{2, 3},
		},
		{
			1,
			[]float64{5, 4},
		},
		{
			2,
			[]float64{9, 6},
		},
		{
			3,
			[]float64{4, 7},
		},
	}
	ballTree := core.NewBallTree(vecs)
	vecs1, err := ballTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), len(vecs))
	ballTree1 := &BallTree{}
	vecs1, err = ballTree1.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 0)
}

func TestBallTreeKNearest(t *testing.T) {
	const numVectors = 50000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 20
	const k = 30

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	ballTree := core.NewBallTree(vecs)

	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	result, err := ballTree.KNearest(query, k)
	assert.Nil(t, err)

	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)

	for i, vec := range result {
		assert.Equal(t, expected[i].ID, vec.ID)
	}
}

func BenchmarkBallTreeKNearest(b *testing.B) {
	const numVectors = 100_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 128
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	ballTree := core.NewBallTree(vecs)

	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ballTree.KNearest(query, k)
	}
}
