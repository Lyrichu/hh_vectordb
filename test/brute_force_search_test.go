package test

import (
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"hh_vectordb/core"
	"math/rand"
	"testing"
	"time"
)

type BruteForceSearch = core.BruteForceSearch

func TestNewBruteForceSearch(t *testing.T) {
	vecs := []Vector{
		{
			0,
			[]float64{2, 3, 1},
		},
		{
			1,
			[]float64{5, 4, 2},
		},
		{
			2,
			[]float64{9, 6, 3},
		},
		{
			3,
			[]float64{4, 7, 4},
		},
		{
			4,
			[]float64{8, 1, 7},
		},
		{
			5,
			[]float64{7, 2, 10},
		},
	}
	bs := core.NewBruteForceSearch(vecs)
	assert.NotNil(t, bs)
}

func TestBruteForceInsert(t *testing.T) {
	bs := &BruteForceSearch{}
	err := bs.Insert(Vector{1, []float64{1, 2, 3, 5}})
	assert.Nil(t, err)
}

func TestBruteForceNearestV1(t *testing.T) {
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
	bs := core.NewBruteForceSearch(vecs)
	for _, vec := range vecs {
		res, err := bs.Nearest(vec)
		assert.Nil(t, err)
		assert.True(t, vec.Equals(res))
	}
}

func TestBruteForceNearestV2(t *testing.T) {
	// 使用大规模随机向量来进行测试
	rand.Seed(time.Now().UnixNano()) // 初始化随机数种子
	// 初始化一些基本参数
	numVectors := 20000
	vecDim := 50
	minValue := -1.0
	maxValue := 10.0

	bs := BruteForceSearch{}
	vectors := make([]Vector, numVectors)

	// 插入随机向量到 KDTree
	for i := 0; i < numVectors; i++ {
		vectors[i] = basic.GenerateRandomVector(int64(i), vecDim, minValue, maxValue)
		err := bs.Insert(vectors[i])
		assert.Nil(t, err)
	}

	// 对于每个向量，查找最近的向量并验证其正确性
	for _, vec := range vectors {
		nearestVec, err := bs.Nearest(vec)
		assert.Nil(t, err)
		assert.True(t, vec.Equals(nearestVec))
		assert.Equal(t, vec.ID, nearestVec.ID)
	}

}

func TestBruteForceDelete(t *testing.T) {
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
	bs := core.NewBruteForceSearch(vecs)
	vecs1, err := bs.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 6)
	err = bs.Delete(Vector{4, []float64{8, 1}})
	assert.Nil(t, err)
	vecs1, err = bs.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)
}

func TestBruteForceVectors(t *testing.T) {
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
	bs := core.NewBruteForceSearch(vecs)
	vecs1, err := bs.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), len(vecs))
	bs1 := &BruteForceSearch{}
	vecs1, err = bs1.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 0)
}

func BenchmarkBruteForceKNearest(b *testing.B) {
	const numVectors = 100_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 50
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 BruteForceSearch 并插入向量
	bs := core.NewBruteForceSearch(vecs)

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询,同时进行基准测试
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = bs.KNearest(query, k)
	}
}

func TestBruteForceInsertBatch(t *testing.T) {
	bs := &BruteForceSearch{}
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
	err := bs.InsertBatch(vecs)
	assert.Nil(t, err)
	vecs1, err := bs.Vectors()
	assert.Equal(t, len(vecs1), len(vecs))
}

func TestBruteForceDeleteBatch(t *testing.T) {
	bs := &BruteForceSearch{}
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
	err := bs.InsertBatch(vecs)
	assert.Nil(t, err)
	vecs1 := []Vector{
		{
			0,
			[]float64{2, 3},
		},
		{
			1,
			[]float64{5, 4},
		},
	}
	err = bs.DeleteBatch(vecs1)
	assert.Nil(t, err)
	resVecs, _ := bs.Vectors()
	assert.Equal(t, len(resVecs), 4)
}

func TestBruteForceSearchWithinRange(t *testing.T) {
	bs := &BruteForceSearch{}
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
	err := bs.InsertBatch(vecs)
	assert.Nil(t, err)
	query := Vector{ID: 6, Values: []float64{4.1, 7.1}}
	res, err := bs.SearchWithinRange(query, 0.2)
	assert.Equal(t, len(res), 1)
	assert.True(t, res[0].Equals(vecs[3]))
}

func TestBruteForceSaveToFile(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 50

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 BruteForceSearch 并插入向量
	bs := core.NewBruteForceSearch(vecs)
	saveFilePath := "/Users/huchengchun/Downloads/hh_vec_db_save01"
	err := bs.SaveToFile(saveFilePath)
	assert.Nil(t, err)
}

func TestBruteForceLoadFromFile(t *testing.T) {
	bs := &core.BruteForceSearch{}
	const numVectors = 10_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 50
	saveFilePath := "/Users/huchengchun/Downloads/hh_vec_db_save01"
	err := bs.LoadFromFile(saveFilePath)
	assert.Nil(t, err)
	vecs, err := bs.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs), numVectors)
	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)
	knVecs, err := bs.KNearest(query, 1000)
	assert.Nil(t, err)
	assert.Equal(t, 1000, len(knVecs))
}
