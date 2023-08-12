package basic

import (
	"fmt"
	"math"
	"strings"
)

type Vector struct {
	ID     int64
	Values []float64
}

const epsilon = 1e-9

func (v Vector) Equals(other Vector) bool {
	if len(v.Values) != len(other.Values) {
		return false
	}
	for i, val := range v.Values {
		if !floatEquals(val, other.Values[i]) {
			return false
		}
	}
	return true
}

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func (v Vector) String() string {
	stringValues := make([]string, len(v.Values))
	for i, value := range v.Values {
		stringValues[i] = fmt.Sprintf("%.2f", value) // 我们选择保留两位小数，但这可以根据需要进行调整
	}
	return fmt.Sprintf("Vector{ID:%v,Values:[%v]}", v.ID, strings.Join(stringValues, ", "))
}

type Item struct {
	Value    Vector
	Distance float64
	Index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].Distance > pq[j].Distance
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.Index = -1
	*pq = old[0 : n-1]
	return item
}
