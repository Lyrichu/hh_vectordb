package core

// kd-tree 相关实现

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"hh_vectordb/basic"
	"math"
	"os"
)

type PriorityQueue = basic.PriorityQueue
type Item = basic.Item

type KDNode struct {
	Vector   Vector
	Left     *KDNode
	Right    *KDNode
	Axis     int
	Distance float64
}

type KDTree struct {
	Root *KDNode
}

func NewKDTree(vectors []Vector) *KDTree {
	tree := &KDTree{}
	for _, vec := range vectors {
		err := tree.Insert(vec)
		if err != nil {
			return nil
		}
	}
	return tree
}

// Insert
//
//	@Description: kd-tree 插入操作
//	@receiver tree kd-tree
//	@param vec 插入向量
//	@return error
func (tree *KDTree) Insert(vec Vector) error {
	tree.Root = insertRecursively(tree.Root, vec, 0)
	return nil
}

// insertRecursively
//
//	@Description: kd-tree 递归插入 vector
//	@param node 待插入node
//	@param vec 需要插入的 vector
//	@param axis 插入维度
//	@return *KDNode
func insertRecursively(node *KDNode, vec Vector, axis int) *KDNode {
	if node == nil {
		return &KDNode{Vector: vec, Axis: axis}
	}

	// 比较轴上的值，如果小于则插入左子树，否则插入右子树
	if vec.Values[axis] < node.Vector.Values[axis] {
		node.Left = insertRecursively(node.Left, vec, (axis+1)%len(vec.Values))
	} else {
		node.Right = insertRecursively(node.Right, vec, (axis+1)%len(vec.Values))
	}

	return node
}

// Nearest
//
//	@Description: 查询最近邻
//	@receiver tree kd-tree
//	@param query 待查询向量
//	@return Vector 查询出的最近邻向量
//	@return error
func (tree *KDTree) Nearest(query Vector) (Vector, error) {
	nearestNode := nearest(tree.Root, query, nil)
	if nearestNode == nil {
		return Vector{}, fmt.Errorf("no nearest neighbor found")
	}
	return nearestNode.Vector, nil
}

// nearest
//
//	@Description: 内部方法,查询最近邻
//	@param node 查询 kd-node
//	@param query 待查询向量
//	@param best 目前最优节点
//	@return *KDNode 目前查询的最优节点
func nearest(node *KDNode, query Vector, best *KDNode) *KDNode {
	if node == nil {
		return best
	}

	// 计算当前节点与查询向量的距离
	d := basic.EuclidDistance(node.Vector.Values, query.Values)

	if best == nil || d < best.Distance {
		best = node
		best.Distance = d
	}

	// 根据当前轴和查询向量的值决定搜索方向
	var next, opposite *KDNode
	if query.Values[node.Axis] < node.Vector.Values[node.Axis] {
		next = node.Left
		opposite = node.Right
	} else {
		next = node.Right
		opposite = node.Left
	}

	best = nearest(next, query, best)

	// 检查对面的子树是否有更接近的点
	if math.Abs(query.Values[node.Axis]-node.Vector.Values[node.Axis]) < best.Distance {
		best = nearest(opposite, query, best)
	}

	return best
}

// Vectors
//
//	@Description: 返回 kd-tree 中当前 vectors
//	@receiver tree
//	@return []Vector
//	@return error
func (tree *KDTree) Vectors() ([]Vector, error) {
	var result []Vector
	tree.collectVectors(tree.Root, &result)
	return result, nil
}

// collectVectors
//
//	@Description: 递归收集 kd-tree 中的 vectors
//	@receiver tree
//	@param node
//	@param vectors
func (tree *KDTree) collectVectors(node *KDNode, vectors *[]Vector) {
	if node == nil {
		return
	}

	*vectors = append(*vectors, node.Vector)

	// 递归遍历左子树和右子树
	tree.collectVectors(node.Left, vectors)
	tree.collectVectors(node.Right, vectors)
}

// Delete
//
//	@Description: kd-tree 删除 向量操作
//	@receiver tree kd-tree
//	@param vec 待删除向量
//	@return error
func (tree *KDTree) Delete(vec Vector) error {
	var deleted bool
	tree.Root, deleted = deleteRecursively(tree.Root, vec, 0)
	if !deleted {
		return fmt.Errorf("vector not found")
	}
	return nil
}

// deleteRecursively
//
//	@Description: 内部方法,kd-tree 执行递归删除
//	@param node kd-node
//	@param vec 待删除向量
//	@param axis 维度
//	@return *KDNode
//	@return bool 是否删除成功
func deleteRecursively(node *KDNode, vec Vector, axis int) (*KDNode, bool) {
	if node == nil {
		return nil, false
	}

	deleted := false

	if node.Vector.Equals(vec) {
		if node.Right != nil {
			minNode := findMin(node.Right, axis, (axis+1)%len(vec.Values))
			node.Vector = minNode.Vector
			node.Right, deleted = deleteRecursively(node.Right, minNode.Vector, (axis+1)%len(vec.Values))
		} else if node.Left != nil {
			return node.Left, true
		} else {
			return nil, true
		}
	} else if vec.Values[axis] < node.Vector.Values[axis] {
		node.Left, deleted = deleteRecursively(node.Left, vec, (axis+1)%len(vec.Values))
	} else {
		node.Right, deleted = deleteRecursively(node.Right, vec, (axis+1)%len(vec.Values))
	}

	return node, deleted
}

// findMin
//
//	@Description: 内部方法,查找最近节点
//	@param node
//	@param axis
//	@param depthAxis
//	@return *KDNode
func findMin(node *KDNode, axis, depthAxis int) *KDNode {
	if node == nil {
		return nil
	}

	if axis == depthAxis {
		if node.Left == nil {
			return node
		}
		return findMin(node.Left, axis, (depthAxis+1)%len(node.Vector.Values))
	}

	leftMin := findMin(node.Left, axis, (depthAxis+1)%len(node.Vector.Values))
	rightMin := findMin(node.Right, axis, (depthAxis+1)%len(node.Vector.Values))

	minNode := node
	if leftMin != nil && leftMin.Vector.Values[axis] < minNode.Vector.Values[axis] {
		minNode = leftMin
	}
	if rightMin != nil && rightMin.Vector.Values[axis] < minNode.Vector.Values[axis] {
		minNode = rightMin
	}

	return minNode
}

// KNearest
//
//	@Description: kd-tree 求 k-近邻向量
//	@receiver tree kd-tree
//	@param query 待查询向量
//	@param k top-k
//	@return []Vector 求解的k-近邻向量
//	@return error
func (tree *KDTree) KNearest(query Vector, k int) ([]Vector, error) {
	pq := make(PriorityQueue, 0, k)
	heap.Init(&pq)

	tree.kNearest(tree.Root, query, 0, k, &pq)

	result := make([]Vector, 0, k)
	for len(pq) > 0 {
		item := heap.Pop(&pq).(*Item)
		result = append(result, item.Value)
	}
	// 反转 result,使得 k-近邻的结果是有序的
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}
	return result, nil
}

// kNearest
//
//	@Description: 内部方法,递归求解 kd-tree 的 k-近邻向量
//	@receiver tree
//	@param node
//	@param query
//	@param axis
//	@param k
//	@param pq
func (tree *KDTree) kNearest(node *KDNode, query basic.Vector, axis, k int, pq *PriorityQueue) {
	if node == nil {
		return
	}

	dist := basic.EuclidDistanceVec(query, node.Vector)

	if len(*pq) < k || dist < (*pq)[0].Distance {
		if len(*pq) == k {
			heap.Pop(pq)
		}
		heap.Push(pq, &Item{
			Value:    node.Vector,
			Distance: dist,
		})
	}

	// Determine which side of the plane the point is in
	nextBranch := node.Left
	otherBranch := node.Right
	if query.Values[axis] > node.Vector.Values[axis] {
		nextBranch = node.Right
		otherBranch = node.Left
	}

	tree.kNearest(nextBranch, query, (axis+1)%len(query.Values), k, pq)

	// Check if other side of plane could have closer points
	if len(*pq) < k || math.Abs(node.Vector.Values[axis]-query.Values[axis]) < (*pq)[0].Distance {
		tree.kNearest(otherBranch, query, (axis+1)%len(query.Values), k, pq)
	}
}

func (tree *KDTree) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := tree.Insert(vec); err != nil {
			return err
		}
	}
	return nil
}

func (tree *KDTree) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := tree.Delete(vec); err != nil {
			return err
		}
	}
	return nil
}

func (tree *KDTree) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	var result []Vector
	tree.collectInRange(tree.Root, query, radius, &result)
	return result, nil
}

func (tree *KDTree) collectInRange(node *KDNode, query Vector, radius float64, vectors *[]Vector) {
	if node == nil {
		return
	}

	dist := basic.EuclidDistanceVec(query, node.Vector)
	if dist <= radius {
		*vectors = append(*vectors, node.Vector)
	}

	if node.Vector.Values[node.Axis]-radius <= query.Values[node.Axis] {
		tree.collectInRange(node.Left, query, radius, vectors)
	}

	if node.Vector.Values[node.Axis]+radius >= query.Values[node.Axis] {
		tree.collectInRange(node.Right, query, radius, vectors)
	}
}

func (tree *KDTree) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(tree.Root)
}

func (tree *KDTree) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(&tree.Root)
}
