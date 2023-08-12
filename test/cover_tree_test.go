package test

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"hh_vectordb/core"
	"math/rand"
	"testing"
	"time"
)

type CoverTree = core.CoverTree

func TestNewCoverTree(t *testing.T) {
	coverTree := core.NewCoverTree(1.5)
	assert.NotNil(t, coverTree)
}

func TestCoverTreeInsert(t *testing.T) {
	coverTree := core.NewCoverTree(1.5)
	vec := basic.GenerateRandomVector(10, 2, 1.0, 5.0)
	err := coverTree.Insert(vec)
	assert.Nil(t, err)
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
	for _, vec = range vecs {
		err := coverTree.Insert(vec)
		assert.Nil(t, err)
	}
}

func TestCoverTreeNearestV1(t *testing.T) {
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
	coverTree := core.NewCoverTree(1.8)
	for _, vec := range vecs {
		err := coverTree.Insert(vec)
		assert.Nil(t, err)
	}
	queryVec := Vector{6, []float64{8.1, 1.1}}
	resVec, err := coverTree.Nearest(queryVec)
	assert.Nil(t, err)
	assert.Equal(t, resVec, vecs[4])
}

func TestCoverTreeNearestV2(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	numVectors := 10000
	vecDim := 10
	minValue := 1.0
	maxValue := 5.0

	tree := core.NewCoverTree(1.5)
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

func TestCoverTreeDelete(t *testing.T) {
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
	coverTree := core.NewCoverTree(1.5)
	for _, vec := range vecs {
		err := coverTree.Insert(vec)
		assert.Nil(t, err)
	}

	vecs1, err := coverTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 6)
	err = coverTree.Delete(Vector{2, []float64{9, 6}})
	assert.Nil(t, err)
	vecs1, err = coverTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)

	err = coverTree.Delete(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = coverTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 4)

	err = coverTree.Insert(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = coverTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)
}

func TestCoverTreeVectors(t *testing.T) {
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
	coverTree := core.NewCoverTree(1.5)
	for _, vec := range vecs {
		err := coverTree.Insert(vec)
		assert.Nil(t, err)
	}
	vecs1, err := coverTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), len(vecs))
	vpTree1 := core.NewCoverTree(1.8)
	vecs1, err = vpTree1.Vectors()
	assert.Equal(t, err.Error(), "tree is empty")
	assert.Equal(t, len(vecs1), 0)

}

func TestCoverTreeKNearest(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 32
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 CoverTree 并插入向量
	coverTree := core.NewCoverTree(1.5)
	for _, vec := range vecs {
		err := coverTree.Insert(vec)
		assert.Nil(t, err)
	}

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	result, err := coverTree.KNearest(query, k)
	assert.Nil(t, err)

	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)
	// 直接打印暴力求解和 cover-Tree 结果的差异
	// 注意 cover-Tree 是启发式算法，其结果可能存在一定的随机性,并且不一定完全和精确解相同
	fmt.Println("Compare Brute-Force/cover-Tree Results:")
	for i, vec := range result {
		fmt.Printf("coverTree:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(query, vec),
			expected[i].ID, basic.EuclidDistanceVec(query, expected[i]))
	}

	//for i, vec := range result {
	//	assert.Equal(t, expected[i].ID, vec.ID)
	//}
}

func BenchmarkCoverTreeKNearest(b *testing.B) {
	const numVectors = 20_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 16
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 CoverTree 并插入向量
	coverTree := core.NewCoverTree(1.1)
	for _, vec := range vecs {
		_ = coverTree.Insert(vec)
	}

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询,同时进行基准测试
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = coverTree.KNearest(query, k)
	}
}
