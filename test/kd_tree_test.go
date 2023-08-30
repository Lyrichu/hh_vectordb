package test

import (
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"hh_vectordb/core"
	"math/rand"
	"testing"
	"time"
)

type KDTree = core.KDTree
type KDNode = core.KDNode

func TestNewKDTree(t *testing.T) {
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
	kdTree := core.NewKDTree(vecs)
	assert.NotNil(t, kdTree)
}

func TestKDTreeInsert(t *testing.T) {
	kdTree := core.NewKDTree([]Vector{})
	vec := Vector{0, []float64{1.1, 2.2, 3.0, 4.1}}
	res := kdTree.Insert(vec)
	assert.Nil(t, res)
}

func TestKDTreeNearestV1(t *testing.T) {
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
	kdTree := core.NewKDTree(vecs)
	queryVec := Vector{6, []float64{8.1, 1.1}}
	resVec, err := kdTree.Nearest(queryVec)
	assert.Nil(t, err)
	assert.Equal(t, resVec, vecs[4])
}

func TestKDTreeNearestV2(t *testing.T) {
	// 使用大规模随机向量来进行测试
	rand.Seed(time.Now().UnixNano()) // 初始化随机数种子
	// 初始化一些基本参数
	numVectors := 10000
	vecDim := 10
	minValue := 1.0
	maxValue := 5.0

	tree := &KDTree{}
	vectors := make([]Vector, numVectors)

	// 插入随机向量到 KDTree
	for i := 0; i < numVectors; i++ {
		vectors[i] = basic.GenerateRandomVector(int64(i), vecDim, minValue, maxValue)
		err := tree.Insert(vectors[i])
		assert.Nil(t, err)
	}

	// 对于每个向量，查找最近的向量并验证其正确性
	for _, vec := range vectors {
		nearestVec, err := tree.Nearest(vec)
		assert.Nil(t, err)
		assert.True(t, vec.Equals(nearestVec))
		assert.Equal(t, vec.ID, nearestVec.ID)
	}
}

func TestKDTreeDelete(t *testing.T) {
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
	kdTree := core.NewKDTree(vecs)
	vecs1, err := kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 6)
	err = kdTree.Delete(Vector{2, []float64{9, 6}})
	assert.Nil(t, err)
	vecs1, err = kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)

	err = kdTree.Delete(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 4)

	err = kdTree.Insert(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)
}

func TestKDTreeVectors(t *testing.T) {
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
	kdTree := core.NewKDTree(vecs)
	vecs1, err := kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), len(vecs))
	kdTree1 := &KDTree{}
	vecs1, err = kdTree1.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 0)

}

func TestKDTreeKNearest(t *testing.T) {
	const numVectors = 1000
	const minValue = 1.0
	const maxValue = 20.0
	const dim = 5
	const k = 10

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 KDTree 并插入向量
	kdTree := core.NewKDTree(vecs)

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询
	result, err := kdTree.KNearest(query, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)

	// 验证 KNearest 的结果
	for i, vec := range result {
		assert.Equal(t, expected[i].ID, vec.ID)
	}
}

func BenchmarkKDTreeKNearest(b *testing.B) {
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

	// 创建 KDTree 并插入向量
	kdTree := core.NewKDTree(vecs)

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询,同时进行基准测试
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = kdTree.KNearest(query, k)
	}
}

func TestBatchInsertAndDelete(t *testing.T) {
	kdTree := core.NewKDTree([]Vector{})

	// BatchInsert
	batchVecs := []Vector{
		{0, []float64{1, 2}},
		{1, []float64{3, 4}},
		{2, []float64{5, 6}},
	}
	err := kdTree.InsertBatch(batchVecs)
	assert.Nil(t, err)
	// Verify insertion
	for _, vec := range batchVecs {
		nearestVec, _ := kdTree.Nearest(vec)
		assert.Equal(t, vec.ID, nearestVec.ID)
	}

	// BatchDelete
	err = kdTree.DeleteBatch(batchVecs)
	assert.Nil(t, err)
	resVecs, err := kdTree.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), 0)
	// Verify deletion
	for _, vec := range batchVecs {
		_, err := kdTree.Nearest(vec)
		assert.Error(t, err) // Expect error as the vector doesn't exist anymore
	}
}

func TestRangeSearch(t *testing.T) {
	kdTree := core.NewKDTree([]Vector{})
	vecs := []Vector{
		{0, []float64{2, 3}},
		{1, []float64{5, 4}},
		{2, []float64{6, 7}},
	}
	for _, v := range vecs {
		err := kdTree.Insert(v)
		assert.Nil(t, err)
	}
	query := Vector{3, []float64{2.01, 3.01}}
	results, err := kdTree.SearchWithinRange(query, 0.1)

	assert.Nil(t, err)
	// We only expect the vector {5, 4} to be in this range
	assert.Equal(t, 1, len(results))
	assert.Equal(t, vecs[0].ID, results[0].ID)
	assert.True(t, results[0].Equals(vecs[0]))
}

func TestKDTreePersistence(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 50
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	kdTree := &KDTree{}
	err := kdTree.InsertBatch(vecs)
	assert.Nil(t, err)
	saveFilePath := "/Users/huchengchun/Downloads/hh_vec_db_save01"
	err = kdTree.SaveToFile(saveFilePath)
	assert.Nil(t, err)

	kdTree = &KDTree{}
	err = kdTree.LoadFromFile(saveFilePath)
	assert.Nil(t, err)

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询
	result, err := kdTree.KNearest(query, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)

	// 验证 KNearest 的结果
	for i, vec := range result {
		assert.Equal(t, expected[i].ID, vec.ID)
	}

}
