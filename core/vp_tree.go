package core

import (
	"container/heap"
	"errors"
	"hh_vectordb/basic"
)

type VPNode struct {
	vantagePoint Vector
	mu           float64
	left         *VPNode
	right        *VPNode
}

type VPTree struct {
	root *VPNode
}

type VPItem struct {
	value    Vector
	priority float64
	index    int
}

type VPPriorityQueue []*VPItem

func (pq VPPriorityQueue) Len() int           { return len(pq) }
func (pq VPPriorityQueue) Less(i, j int) bool { return pq[i].priority > pq[j].priority }
func (pq VPPriorityQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *VPPriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*VPItem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *VPPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

func NewVPTree(vectors []Vector) *VPTree {
	tree := &VPTree{}
	tree.root = tree.buildVPTree(vectors)
	return tree
}

func (tree *VPTree) buildVPTree(vectors []Vector) *VPNode {
	if len(vectors) == 0 {
		return nil
	}

	vp := vectors[0] // For simplicity, choose the first point as the vantage point
	if len(vectors) == 1 {
		return &VPNode{vantagePoint: vp}
	}

	// Calculate the median distance from the vantage point to all other points
	distances := make([]float64, len(vectors)-1)
	for i, v := range vectors[1:] {
		distances[i] = basic.EuclidDistanceVec(vp, v)
	}
	mu := basic.Median(distances)

	var leftSet []Vector
	var rightSet []Vector

	for _, v := range vectors[1:] {
		if basic.EuclidDistanceVec(vp, v) < mu {
			leftSet = append(leftSet, v)
		} else {
			rightSet = append(rightSet, v)
		}
	}

	return &VPNode{
		vantagePoint: vp,
		mu:           mu,
		left:         tree.buildVPTree(leftSet),
		right:        tree.buildVPTree(rightSet),
	}
}

func (tree *VPTree) Nearest(query Vector) (Vector, error) {
	// For simplicity, assume KNearest with k = 1
	results, err := tree.KNearest(query, 1)
	if err != nil || len(results) == 0 {
		return Vector{}, err
	}
	return results[0], nil
}

func (tree *VPTree) Insert(vec Vector) error {
	tree.root = tree.insertRecursive(tree.root, vec)
	return nil
}

func (tree *VPTree) insertRecursive(vpNode *VPNode, vec Vector) *VPNode {
	if vpNode == nil {
		return &VPNode{vantagePoint: vec}
	}
	if basic.EuclidDistanceVec(vec, vpNode.vantagePoint) < vpNode.mu {
		vpNode.left = tree.insertRecursive(vpNode.left, vec)
	} else {
		vpNode.right = tree.insertRecursive(vpNode.right, vec)
	}
	return vpNode
}

func (tree *VPTree) KNearest(query Vector, k int) ([]Vector, error) {
	pq := make(VPPriorityQueue, 0, k)
	heap.Init(&pq)

	tree.kNearestRecursive(tree.root, query, k, &pq)

	results := make([]Vector, len(pq))
	for i := len(pq) - 1; i >= 0; i-- {
		results[i] = heap.Pop(&pq).(*VPItem).value
	}

	return results, nil
}

func (tree *VPTree) kNearestRecursive(VPNode *VPNode, query Vector, k int, pq *VPPriorityQueue) {
	if VPNode == nil {
		return
	}

	d := basic.EuclidDistanceVec(query, VPNode.vantagePoint)

	// Check if the current node's vector is closer than the furthest found so far
	if len(*pq) < k || d < (*pq)[0].priority {
		if len(*pq) == k {
			heap.Pop(pq)
		}
		heap.Push(pq, &VPItem{value: VPNode.vantagePoint, priority: d})
	}

	if d < VPNode.mu {
		tree.kNearestRecursive(VPNode.left, query, k, pq)
		if len(*pq) < k || d+VPNode.mu <= (*pq)[0].priority {
			tree.kNearestRecursive(VPNode.right, query, k, pq)
		}
	} else {
		tree.kNearestRecursive(VPNode.right, query, k, pq)
		if len(*pq) < k || d-VPNode.mu <= (*pq)[0].priority {
			tree.kNearestRecursive(VPNode.left, query, k, pq)
		}
	}

}

func (tree *VPTree) Vectors() ([]Vector, error) {
	vectors := make([]Vector, 0)
	tree.inOrderTraversal(tree.root, &vectors)
	return vectors, nil
}

func (tree *VPTree) inOrderTraversal(VPNode *VPNode, vectors *[]Vector) {
	if VPNode == nil {
		return
	}
	tree.inOrderTraversal(VPNode.left, vectors)
	*vectors = append(*vectors, VPNode.vantagePoint)
	tree.inOrderTraversal(VPNode.right, vectors)
}

func (tree *VPTree) Delete(vec Vector) error {
	success := false
	tree.root, success = tree.deleteRecursive(tree.root, vec)
	if !success {
		return errors.New("vector not found")
	}
	return nil
}

func (tree *VPTree) deleteRecursive(VPNode *VPNode, vec Vector) (*VPNode, bool) {
	if VPNode == nil {
		return nil, false
	}

	// If the current node is the target
	if VPNode.vantagePoint.Equals(vec) {
		// Node with only right child or no child
		if VPNode.left == nil {
			temp := VPNode.right
			VPNode = nil
			return temp, true
		} else if VPNode.right == nil { // Node with only left child
			temp := VPNode.left
			VPNode = nil
			return temp, true
		}

		// Node with two children
		// Get the inorder predecessor (rightmost node in left subtree)
		temp, _ := tree.findMax(VPNode.left)
		VPNode.vantagePoint = temp
		// Delete the inorder predecessor
		VPNode.left, _ = tree.deleteRecursive(VPNode.left, temp)
	} else {
		// If the vector to be deleted is smaller than the VPNode's
		// vector, then it lies in left subtree
		if basic.EuclidDistanceVec(VPNode.vantagePoint, vec) < VPNode.mu {
			VPNode.left, _ = tree.deleteRecursive(VPNode.left, vec)
		} else {
			// Else it lies in right subtree
			VPNode.right, _ = tree.deleteRecursive(VPNode.right, vec)
		}
	}

	return VPNode, true
}

func (tree *VPTree) findMax(VPNode *VPNode) (Vector, bool) {
	if VPNode == nil {
		return Vector{}, false
	}
	if VPNode.right != nil {
		return tree.findMax(VPNode.right)
	}
	return VPNode.vantagePoint, true
}
