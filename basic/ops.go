package basic

import (
	"math"
	"math/rand"
	"sort"
)

// EuclidDistance
//
//	@Description:计算两个向量之间的欧几里得距离
//	@param a 数组a
//	@param b 数组b
//	@return float64 欧几里得距离
func EuclidDistance(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += (a[i] - b[i]) * (a[i] - b[i])
	}
	return math.Sqrt(sum)
}

// EuclidDistanceVec
//
//	@Description: 计算两个向量之间的欧几里得距离
//	@param a 向量 a
//	@param b 向量 b
//	@return float64 欧几里得距离
func EuclidDistanceVec(a, b Vector) float64 {
	return EuclidDistance(a.Values, b.Values)
}

// GenerateRandomVector
//
//	@Description: 生成随机 Vector
//	@param id 向量 ID
//	@param dim 向量维度
//	@param minValue 向量最小值
//	@param maxValue 向量最大值
//	@return Vector
func GenerateRandomVector(id int64, dim int, minValue float64, maxValue float64) Vector {
	values := make([]float64, dim)
	for i := 0; i < dim; i++ {
		values[i] = rand.Float64()*(maxValue-minValue) + minValue
	}
	return Vector{ID: id, Values: values}
}

func Median(nums []float64) float64 {
	sortedNums := make([]float64, len(nums))
	copy(sortedNums, nums)
	sort.Float64s(sortedNums)

	n := len(sortedNums)
	if n%2 == 0 {
		return (sortedNums[n/2-1] + sortedNums[n/2]) / 2
	}
	return sortedNums[n/2]
}
