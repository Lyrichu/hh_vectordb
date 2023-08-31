package test

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"hh_vectordb/basic"
	"hh_vectordb/core"
	"testing"
	"time"
)

func TestPQVectors(t *testing.T) {
	const numVectors = 1_0000
	const minValue = -5.0
	const maxValue = 5.0
	const dim = 50

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 5)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}
	resVecs, err := pq.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), numVectors)
}

func TestPQNearest(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -5.0
	const maxValue = 5.0
	const dim = 30

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 3)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}
	// Test
	queryVec := vecs[10]
	result, err := pq.Nearest(queryVec)
	assert.Nil(t, err)
	assert.Equal(t, result.ID, int64(10))
	assert.True(t, result.Equals(queryVec))
}

func TestPQDelete(t *testing.T) {
	const numVectors = 1_0000
	const minValue = -5.0
	const maxValue = 5.0
	const dim = 50

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 5)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}
	resVecs, err := pq.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), numVectors)
	for i := 0; i < 100; i++ {
		err = pq.Delete(vecs[i])
		assert.Nil(t, err)
	}
	resVecs, err = pq.Vectors()
	assert.Equal(t, len(resVecs), numVectors-100)
}

func TestPQKNearest(t *testing.T) {
	const numVectors = 20_0000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 30
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 15)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}
	// Test
	queryVec := vecs[10]
	resVecs, err := pq.KNearest(queryVec, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(queryVec, k)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(resVecs, expected, false)
	fmt.Println(ratio)
	fmt.Println("Compare Brute-Force/PQ Results:")
	for i, vec := range resVecs {
		fmt.Printf("pq:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(queryVec, vec),
			expected[i].ID, basic.EuclidDistanceVec(queryVec, expected[i]))
	}
}

func TestPQKNearestRefined(t *testing.T) {
	const numVectors = 20_0000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 30
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 15)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}

	queryVec := vecs[10]
	resVecs, err := pq.KNearestRefined(queryVec, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(queryVec, k)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(resVecs, expected, false)
	fmt.Println(ratio)
	fmt.Println("Compare Brute-Force/PQ Results:")
	for i, vec := range resVecs {
		fmt.Printf("pq:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(queryVec, vec),
			expected[i].ID, basic.EuclidDistanceVec(queryVec, expected[i]))
	}
}

func TestPQKNearestConcurrent(t *testing.T) {
	const numVectors = 20_0000
	const minValue = -20.0
	const maxValue = 20.0
	const dim = 30
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 15)
	pq.Train(vecs, 100)

	for _, v := range vecs {
		if err := pq.Insert(v); err != nil {
			t.Fatalf("Failed to insert vector: %v", err)
		}
	}

	queryVec := vecs[10]
	resVecs, err := pq.KNearestConcurrent(queryVec, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(queryVec, k)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(resVecs, expected, false)
	fmt.Println(ratio)
	fmt.Println("Compare Brute-Force/PQ Results:")
	for i, vec := range resVecs {
		fmt.Printf("pq:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(queryVec, vec),
			expected[i].ID, basic.EuclidDistanceVec(queryVec, expected[i]))
	}
}

func TestPQInsertBatch(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -5.0
	const maxValue = 5.0
	const dim = 50

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 5)
	pq.Train(vecs, 100)

	err := pq.InsertBatch(vecs)
	assert.Nil(t, err)
	resVecs, err := pq.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), numVectors)
}

func TestPQDeleteBatch(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -5.0
	const maxValue = 5.0
	const dim = 50

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 5)
	pq.Train(vecs, 100)

	err := pq.InsertBatch(vecs)
	assert.Nil(t, err)
	deleteVecs := vecs[100:1000]
	err = pq.DeleteBatch(deleteVecs)
	assert.Nil(t, err)
	resVecs, err := pq.Vectors()
	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), numVectors-900)
}

func TestPQSearchWithinInterval(t *testing.T) {
	const numVectors = 20_0000
	const minValue = -20.0
	const maxValue = 10.0
	const dim = 50

	const minDist = 10.0
	const maxDis = 500.0

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(10, 5)
	pq.Train(vecs, 100)

	err := pq.InsertBatch(vecs)
	assert.Nil(t, err)
	query := basic.GenerateRandomVector(300000, dim, minValue, maxValue)
	resVecs, err := pq.SearchWithinInterval(query, minDist, maxDis)
	assert.Nil(t, err)
	assert.Greater(t, len(resVecs), 0)
	fmt.Println(resVecs)
}

func TestPQSearchWithinRange(t *testing.T) {
	const numVectors = 10_0000
	const minValue = -20.0
	const maxValue = 10.0
	const dim = 20

	const radius = 50.0

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(5, 10)
	pq.Train(vecs, 100)

	err := pq.InsertBatch(vecs)
	assert.Nil(t, err)
	query := basic.GenerateRandomVector(300000, dim, minValue, maxValue)
	resVecs, err := pq.SearchWithinRange(query, radius)
	assert.Nil(t, err)
	assert.Greater(t, len(resVecs), 0)

	bs := &BruteForceSearch{}
	err = bs.InsertBatch(vecs)
	assert.Nil(t, err)
	expected, err := bs.SearchWithinRange(query, radius)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(resVecs, expected, false)
	fmt.Println(ratio)
	fmt.Printf("len(resVecs) = %v,len(expected) = %v\n", len(resVecs), len(expected))
	fmt.Println("Compare Brute-Force/PQ Results:")
	for i, vec := range resVecs {
		fmt.Printf("pq:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(query, vec),
			expected[i].ID, basic.EuclidDistanceVec(query, expected[i]))
	}
}

func TestPQPersistence(t *testing.T) {
	const numVectors = 500_0000
	const minValue = -20.0
	const maxValue = 10.0
	const dim = 20
	const k = 100

	// 随机生成 numVectors 个向量
	vecs := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vecs[i] = basic.GenerateRandomVector(int64(i), dim, minValue, maxValue)
	}
	pq := core.NewPQ(5, 10)
	pq.Train(vecs, 100)

	err := pq.InsertBatch(vecs)
	assert.Nil(t, err)
	saveFile := "/Users/huchengchun/Downloads/test01.gob"
	err = pq.SaveToFile(saveFile)
	assert.Nil(t, err)

	pq = core.NewPQ(5, 30)
	err = pq.LoadFromFile(saveFile)
	assert.Nil(t, err)
	resVecs, err := pq.Vectors()
	assert.Equal(t, len(resVecs), numVectors)

	queryVec := vecs[10]
	resVecs, err = pq.KNearestRefined(queryVec, k)
	assert.Nil(t, err)

	// 使用暴力方法找到最近的 k 个向量
	bs := core.NewBruteForceSearch(vecs)
	expected, err := bs.KNearest(queryVec, k)
	assert.Nil(t, err)
	ratio := basic.TwoVectorArrIntersectionRatio(resVecs, expected, false)
	fmt.Println(ratio)
	fmt.Println("Compare Brute-Force/PQ Results:")
	for i, vec := range resVecs {
		fmt.Printf("pq:%v,%v -- bruteForce:%v,%v\n", vec.ID, basic.EuclidDistanceVec(queryVec, vec),
			expected[i].ID, basic.EuclidDistanceVec(queryVec, expected[i]))
	}

}

func TestPQLoadFromFile(t *testing.T) {

	saveFile := "/Users/huchengchun/Downloads/test01.gob"

	pq := core.NewPQ(5, 10)
	err := pq.LoadFromFile(saveFile)
	assert.Nil(t, err)
	fmt.Println("finish load....")
	resVecs, err := pq.Vectors()
	assert.Greater(t, len(resVecs), 10000)

	queryVec := basic.GenerateRandomVector(int64(10000000), 20, -20.0, 10.0)

	// Start the timer
	startTime := time.Now()

	resVecs, err = pq.KNearestConcurrent(queryVec, 10)

	// Calculate and print the elapsed time
	elapsedTime := time.Since(startTime)
	fmt.Printf("kNearest took %v to execute.\n", elapsedTime)

	assert.Nil(t, err)
	assert.Equal(t, len(resVecs), 10)
}
