package core

import (
	"container/heap"
	"errors"
	"hh_vectordb/basic"
)

type VectorDistance struct {
	vec  Vector
	dist float64
}

type DistanceHeap []VectorDistance

func (h DistanceHeap) Len() int           { return len(h) }
func (h DistanceHeap) Less(i, j int) bool { return h[i].dist > h[j].dist } // We want a max-heap
func (h DistanceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *DistanceHeap) Push(x interface{}) {
	*h = append(*h, x.(VectorDistance))
}

func (h *DistanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type BallTree struct {
	center  Vector
	radius  float64
	left    *BallTree
	right   *BallTree
	isLeaf  bool
	payload Vector
}

func NewBallTree(vectors []Vector) *BallTree {
	if len(vectors) == 0 || vectors == nil {
		return &BallTree{
			isLeaf:  true,
			payload: Vector{},
		}
	}

	if len(vectors) <= 1 {
		var payload Vector
		if len(vectors) == 1 {
			payload = vectors[0]
		}
		return &BallTree{
			isLeaf:  true,
			payload: payload,
		}
	}

	center, radius := computeBoundingSphere(vectors)
	left, right := splitV1(vectors)

	// Check if split is working correctly
	if len(left) == 0 || len(right) == 0 {
		panic("split function is not splitting the vectors correctly")
	}

	return &BallTree{
		center: center,
		radius: radius,
		left:   NewBallTree(left),
		right:  NewBallTree(right),
	}
}

func computeBoundingSphere(vectors []Vector) (Vector, float64) {
	if len(vectors) == 0 {
		return Vector{}, 0.0 // Return a default vector and radius of 0
	}

	var center Vector
	dim := len(vectors[0].Values)
	center.Values = make([]float64, dim)

	for _, v := range vectors {
		for i, val := range v.Values {
			center.Values[i] += val
		}
	}

	for i := range center.Values {
		center.Values[i] /= float64(len(vectors))
	}

	maxDist := 0.0
	for _, v := range vectors {
		dist := basic.EuclidDistanceVec(center, v)
		if dist > maxDist {
			maxDist = dist
		}
	}

	return center, maxDist
}

func splitV1(vectors []Vector) ([]Vector, []Vector) {
	if len(vectors) < 2 {
		return vectors, []Vector{}
	}

	// For simplicity, splitting at the median of the first dimension
	pivot := vectors[0].Values[0]

	var left, right []Vector
	for _, v := range vectors[1:] { // start from the second element to ensure pivot is not compared
		if v.Values[0] < pivot {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}

	// Ensure that the pivot is added to one of the slices.
	// This prevents both slices from being empty.
	if len(left) < len(right) {
		left = append(left, vectors[0])
	} else {
		right = append(right, vectors[0])
	}

	return left, right
}

func (tree *BallTree) Insert(vec Vector) error {
	if tree.isLeaf && tree.payload.Values == nil { // the tree is empty
		tree.payload = vec
		return nil
	}

	if tree.isLeaf {
		left, right := splitV1([]Vector{tree.payload, vec})

		if len(left) == 0 || len(right) == 0 {
			// Handle case when split doesn't return valid left/right children
			return errors.New("failed to split the vectors properly")
		}

		tree.isLeaf = false
		tree.left = NewBallTree(nil)
		tree.right = NewBallTree(nil)

		err := tree.left.Insert(left[0])
		if err != nil {
			return err
		}
		return tree.right.Insert(right[0])
	}

	if basic.EuclidDistanceVec(tree.center, vec) <= tree.radius {
		if tree.left == nil {
			tree.left = NewBallTree(nil)
		}
		return tree.left.Insert(vec)
	} else {
		if tree.right == nil {
			tree.right = NewBallTree(nil)
		}
		return tree.right.Insert(vec)
	}
}

func (tree *BallTree) Nearest(query Vector) (Vector, error) {
	if tree.isLeaf {
		return tree.payload, nil
	}

	distToLeft := basic.EuclidDistanceVec(tree.left.center, query) - tree.left.radius
	distToRight := basic.EuclidDistanceVec(tree.right.center, query) - tree.right.radius

	if distToLeft < distToRight {
		closest, _ := tree.left.Nearest(query)
		other, _ := tree.right.Nearest(query)

		if basic.EuclidDistanceVec(query, closest) < basic.EuclidDistanceVec(query, other) {
			return closest, nil
		}
		return other, nil
	}

	closest, _ := tree.right.Nearest(query)
	other, _ := tree.left.Nearest(query)

	if basic.EuclidDistanceVec(query, closest) < basic.EuclidDistanceVec(query, other) {
		return closest, nil
	}
	return other, nil
}

func (tree *BallTree) Vectors() ([]Vector, error) {
	if tree == nil {
		return nil, errors.New("tree is nil")
	}

	if tree.isLeaf {
		return []Vector{tree.payload}, nil
	}

	var leftVectors []Vector
	if tree.left != nil {
		var err error
		leftVectors, err = tree.left.Vectors()
		if err != nil {
			return nil, err
		}
	}

	var rightVectors []Vector
	if tree.right != nil {
		var err error
		rightVectors, err = tree.right.Vectors()
		if err != nil {
			return nil, err
		}
	}

	return append(leftVectors, rightVectors...), nil
}

func (tree *BallTree) Delete(vec Vector) error {
	if tree == nil {
		return errors.New("tree is nil")
	}

	// Check if we're at a leaf node.
	if tree.isLeaf {
		if tree.payload.Equals(vec) {
			// This is the vector to delete.
			tree.payload = Vector{} // Reset the payload.
			tree.isLeaf = false     // Mark the tree as non-leaf, making it effectively empty.
			return nil
		}
		return errors.New("vector not found")
	}

	// Try to delete from the left subtree.
	err := tree.left.Delete(vec)
	if err == nil {
		return nil // If to delete was successful in the left tree, return.
	}

	// If not found in left subtree, try the right subtree.
	err = tree.right.Delete(vec)
	if err == nil {
		return nil // If to delete was successful in the right tree, return.
	}

	// If we reach here, the vector wasn't found in either subtree.
	return errors.New("vector not found")
}

func (tree *BallTree) KNearest(query Vector, k int) ([]Vector, error) {
	if k <= 0 {
		return nil, errors.New("k should be greater than 0")
	}

	h := &DistanceHeap{}
	heap.Init(h)
	tree.kNearestRecursive(query, k, h)

	vectors := make([]Vector, 0, k)
	for h.Len() > 0 {
		vectors = append(vectors, heap.Pop(h).(VectorDistance).vec)
	}

	// Reverse to get vectors in increasing order of distance
	for i, j := 0, len(vectors)-1; i < j; i, j = i+1, j-1 {
		vectors[i], vectors[j] = vectors[j], vectors[i]
	}

	return vectors, nil
}

func (tree *BallTree) kNearestRecursive(query Vector, k int, h *DistanceHeap) {
	if tree.isLeaf {
		dist := basic.EuclidDistanceVec(tree.payload, query)
		if h.Len() < k || dist < (*h)[0].dist {
			heap.Push(h, VectorDistance{tree.payload, dist})
		}
		if h.Len() > k {
			heap.Pop(h)
		}
		return
	}

	distToLeft := basic.EuclidDistanceVec(tree.left.center, query) - tree.left.radius
	distToRight := basic.EuclidDistanceVec(tree.right.center, query) - tree.right.radius

	// Recur to the closer child first
	if distToLeft < distToRight {
		tree.left.kNearestRecursive(query, k, h)
		if h.Len() < k || distToRight < (*h)[0].dist {
			tree.right.kNearestRecursive(query, k, h)
		}
	} else {
		tree.right.kNearestRecursive(query, k, h)
		if h.Len() < k || distToLeft < (*h)[0].dist {
			tree.left.kNearestRecursive(query, k, h)
		}
	}
}

func splitV2(vectors []Vector) ([]Vector, []Vector) {
	if len(vectors) < 2 {
		return vectors, []Vector{}
	}

	// Determine dimension with the highest variance.
	dimension, _ := maxVarianceDimension(vectors)

	// Use QuickSelect to find the median of the selected dimension.
	median := quickSelect(vectors, len(vectors)/2, dimension)

	var left, right []Vector
	for _, v := range vectors {
		if v.Values[dimension] < median {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}

	// Check for empty slices and handle them
	if len(left) == 0 || len(right) == 0 {
		mid := len(vectors) / 2
		return vectors[:mid], vectors[mid:]
	}

	return left, right
}

func maxVarianceDimension(vectors []Vector) (int, float64) {
	if len(vectors) == 0 {
		return 0, 0.0
	}

	dim := len(vectors[0].Values)
	var maxVar float64
	var maxDim int

	for d := 0; d < dim; d++ {
		mean := 0.0
		for _, v := range vectors {
			mean += v.Values[d]
		}
		mean /= float64(len(vectors))

		variance := 0.0
		for _, v := range vectors {
			diff := v.Values[d] - mean
			variance += diff * diff
		}
		variance /= float64(len(vectors))

		if variance > maxVar {
			maxVar = variance
			maxDim = d
		}
	}

	return maxDim, maxVar
}

func quickSelect(vectors []Vector, k int, dimension int) float64 {
	if len(vectors) <= k {
		return vectors[len(vectors)-1].Values[dimension]
	}

	pivot := vectors[k].Values[dimension]
	low, high := []Vector{}, []Vector{}

	for _, v := range vectors {
		if v.Values[dimension] < pivot {
			low = append(low, v)
		} else if v.Values[dimension] > pivot {
			high = append(high, v)
		}
	}

	if k < len(low) {
		return quickSelect(low, k, dimension)
	} else if k > len(vectors)-len(high) {
		return quickSelect(high, k-(len(vectors)-len(high)), dimension)
	}
	return pivot
}
