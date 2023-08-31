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

func TestNewLSH(t *testing.T) {
	lsh := core.NewLSH(10, 10)
	assert.NotNil(t, lsh)
}

func TestLSHInsert(t *testing.T) {
	lsh := core.NewLSH(10, 10)
	vec := Vector{20, []float64{2.2, 3.0}}
	err := lsh.Insert(vec)
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
	for _, vec := range vecs {
		err = lsh.Insert(vec)
		assert.Nil(t, err)
	}
}

func TestLSHNearestV1(t *testing.T) {
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
	lsh := core.NewLSH(10, 10)
	for _, vec := range vecs {
		err := lsh.Insert(vec)
		assert.Nil(t, err)
	}
	queryVec := Vector{6, []float64{8.1, 1.1}}
	resVec, err := lsh.Nearest(queryVec)
	assert.Nil(t, err)
	assert.Equal(t, resVec, vecs[4])
}

func TestLSHNearestV2(t *testing.T) {
	// 使用大规模随机向量来进行测试
	rand.Seed(time.Now().UnixNano()) // 初始化随机数种子
	// 初始化一些基本参数
	numVectors := 10000
	vecDim := 10
	minValue := 1.0
	maxValue := 5.0

	lsh := core.NewLSH(1000, 3000)
	vectors := make([]Vector, numVectors)

	// 插入随机向量到 KDTree
	for i := 0; i < numVectors; i++ {
		vectors[i] = basic.GenerateRandomVector(int64(i), vecDim, minValue, maxValue)
		err := lsh.Insert(vectors[i])
		assert.Nil(t, err)
	}

	// 对于每个向量，查找最近的向量并验证其正确性
	for _, vec := range vectors {
		nearestVec, err := lsh.Nearest(vec)
		assert.Nil(t, err)
		assert.True(t, vec.Equals(nearestVec))
		assert.Equal(t, vec.ID, nearestVec.ID)
	}
}

func TestLSHDelete(t *testing.T) {
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
	lsh := core.NewLSH(1000, 1000)
	for _, vec := range vecs {
		err := lsh.Insert(vec)
		assert.Nil(t, err)
	}
	vecs1, err := lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 6)
	err = lsh.Delete(Vector{2, []float64{9, 6}})
	assert.Nil(t, err)
	vecs1, err = lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)

	err = lsh.Delete(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 4)

	err = lsh.Insert(Vector{5, []float64{7, 2}})
	assert.Nil(t, err)
	vecs1, err = lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 5)
}

func TestLSHVectors(t *testing.T) {
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
	lsh := core.NewLSH(100, 100)
	for _, vec := range vecs {
		err := lsh.Insert(vec)
		assert.Nil(t, err)
	}
	vecs1, err := lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), len(vecs))
	lsh1 := core.NewLSH(10, 100)
	vecs1, err = lsh1.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(vecs1), 0)

}

func TestLSHKNearest(t *testing.T) {
	const numVectors = 100_0000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 32
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 LSH 并插入向量
	lsh := core.NewLSH(50, 100000)
	for _, vec := range vecs {
		err := lsh.Insert(vec)
		assert.Nil(t, err)
	}
	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	result, err := lsh.KNearest(query, k)
	assert.Nil(t, err)

	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)
	// 直接打印暴力求解和 LSH 结果的差异
	// 注意 LSH 是近似搜索算法，其结果可能存在一定的随机性,并且不一定完全和精确解相同
	fmt.Println("Compare Brute-Force/LSH Results:")
	for i, vec := range result {
		fmt.Printf("lsh:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(query, vec),
			expected[i].ID, basic.EuclidDistanceVec(query, expected[i]))
	}

	//for i, vec := range result {
	//	assert.Equal(t, expected[i].ID, vec.ID)
	//}
}

func BenchmarkLSHKNearest(b *testing.B) {
	const numVectors = 50_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 128
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}

	// 创建 LSH 并插入向量
	lsh := core.NewLSH(100, 10000)
	for _, vec := range vecs {
		_ = lsh.Insert(vec)
	}

	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询,同时进行基准测试
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = lsh.KNearest(query, k)
	}
}

func TestLSHInsertBatch(t *testing.T) {
	lsh := core.NewLSH(100, 10000)
	vecs := []Vector{
		basic.GenerateRandomVector(0, 4, 1.0, 5.0),
		basic.GenerateRandomVector(1, 4, 1.0, 5.0),
		basic.GenerateRandomVector(2, 4, 1.0, 5.0),
	}
	err := lsh.InsertBatch(vecs)
	assert.Nil(t, err)
	resVecs, err := lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), len(vecs))
}

func TestLSHDeleteBatch(t *testing.T) {
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
	}
	lsh := core.NewLSH(100, 10000)
	err := lsh.InsertBatch(vecs)
	assert.Nil(t, err)
	err = lsh.DeleteBatch([]Vector{vecs[0], vecs[2]})
	assert.Nil(t, err)
	resVecs, err := lsh.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, 1, len(resVecs))
}

func TestLSHInRange(t *testing.T) {
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
	}
	lsh := core.NewLSH(100, 10000)
	err := lsh.InsertBatch(vecs)
	assert.Nil(t, err)

	centerVec := Vector{
		99,
		[]float64{5, 5},
	}
	radius := 3.0
	resultVecs, err := lsh.SearchWithinRange(centerVec, radius)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(resultVecs))
	assert.True(t, basic.VectorExistsInSlice(resultVecs[0], vecs))
	assert.True(t, resultVecs[0].Equals(vecs[1]))
}

func TestLSHPersistence(t *testing.T) {
	const numVectors = 5_0000
	const minValue = -10.0
	const maxValue = 10.0
	const dim = 50
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	saveFilePath := "/Users/huchengchun/Downloads/hh_vec_db_save01"
	lsh := core.NewLSH(100, 10000)
	err := lsh.InsertBatch(vecs)
	assert.Nil(t, err)
	err = lsh.SaveToFile(saveFilePath)
	assert.Nil(t, err)

	lsh = core.NewLSH(100, 10000)
	err = lsh.LoadFromFile(saveFilePath)
	assert.Nil(t, err)
	resVecs, err := lsh.Vectors()
	assert.Equal(t, len(resVecs), len(vecs))
	// 随机选择一个查询向量
	query := basic.GenerateRandomVector(int64(numVectors), dim, minValue, maxValue)

	// 使用 KNearest 查询
	result, err := lsh.KNearest(query, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(query, k)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(result, expected, false)
	fmt.Println(ratio)
	fmt.Println("Compare Brute-Force/LSH Results:")
	for i, vec := range result {
		fmt.Printf("lsh:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(query, vec),
			expected[i].ID, basic.EuclidDistanceVec(query, expected[i]))
	}
}
