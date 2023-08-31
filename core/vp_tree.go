package core

import (
	"container/heap"
	"encoding/gob"
	"errors"
	"hh_vectordb/basic"
	"os"
)

type VPNode struct {
	VantagePoint Vector
	Mu           float64
	Left         *VPNode
	Right        *VPNode
}

type VPTree struct {
	Root *VPNode
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
	tree.Root = tree.buildVPTree(vectors)
	return tree
}

func (tree *VPTree) buildVPTree(vectors []Vector) *VPNode {
	if len(vectors) == 0 {
		return nil
	}

	vp := vectors[0] // For simplicity, choose the first point as the vantage point
	if len(vectors) == 1 {
		return &VPNode{VantagePoint: vp}
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
		VantagePoint: vp,
		Mu:           mu,
		Left:         tree.buildVPTree(leftSet),
		Right:        tree.buildVPTree(rightSet),
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
	tree.Root = tree.insertRecursive(tree.Root, vec)
	return nil
}

func (tree *VPTree) insertRecursive(vpNode *VPNode, vec Vector) *VPNode {
	if vpNode == nil {
		return &VPNode{VantagePoint: vec}
	}
	if basic.EuclidDistanceVec(vec, vpNode.VantagePoint) < vpNode.Mu {
		vpNode.Left = tree.insertRecursive(vpNode.Left, vec)
	} else {
		vpNode.Right = tree.insertRecursive(vpNode.Right, vec)
	}
	return vpNode
}

func (tree *VPTree) KNearest(query Vector, k int) ([]Vector, error) {
	pq := make(VPPriorityQueue, 0, k)
	heap.Init(&pq)

	tree.kNearestRecursive(tree.Root, query, k, &pq)

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

	d := basic.EuclidDistanceVec(query, VPNode.VantagePoint)

	// Check if the current node's vector is closer than the furthest found so far
	if len(*pq) < k || d < (*pq)[0].priority {
		if len(*pq) == k {
			heap.Pop(pq)
		}
		heap.Push(pq, &VPItem{value: VPNode.VantagePoint, priority: d})
	}

	if d < VPNode.Mu {
		tree.kNearestRecursive(VPNode.Left, query, k, pq)
		if len(*pq) < k || d+VPNode.Mu <= (*pq)[0].priority {
			tree.kNearestRecursive(VPNode.Right, query, k, pq)
		}
	} else {
		tree.kNearestRecursive(VPNode.Right, query, k, pq)
		if len(*pq) < k || d-VPNode.Mu <= (*pq)[0].priority {
			tree.kNearestRecursive(VPNode.Left, query, k, pq)
		}
	}

}

func (tree *VPTree) Vectors() ([]Vector, error) {
	vectors := make([]Vector, 0)
	tree.inOrderTraversal(tree.Root, &vectors)
	return vectors, nil
}

func (tree *VPTree) inOrderTraversal(VPNode *VPNode, vectors *[]Vector) {
	if VPNode == nil {
		return
	}
	tree.inOrderTraversal(VPNode.Left, vectors)
	*vectors = append(*vectors, VPNode.VantagePoint)
	tree.inOrderTraversal(VPNode.Right, vectors)
}

func (tree *VPTree) Delete(vec Vector) error {
	success := false
	tree.Root, success = tree.deleteRecursive(tree.Root, vec)
	if !success {
		return errors.New("vector not found")
	}
	return nil
}

func (tree *VPTree) deleteRecursive(VPNode *VPNode, vec Vector) (*VPNode, bool) {
	if VPNode == nil {
		return nil, false
	}

	if VPNode.VantagePoint.Equals(vec) {
		vectors, _ := tree.subTreeVectors(VPNode) // Collect all vectors from the subtree
		for i, v := range vectors {
			if v.Equals(vec) {
				// Remove the vector from the slice
				vectors = append(vectors[:i], vectors[i+1:]...)
				break
			}
		}
		return tree.buildVPTree(vectors), true // Rebuild the subtree
	} else {
		if basic.EuclidDistanceVec(VPNode.VantagePoint, vec) < VPNode.Mu {
			VPNode.Left, _ = tree.deleteRecursive(VPNode.Left, vec)
		} else {
			VPNode.Right, _ = tree.deleteRecursive(VPNode.Right, vec)
		}
	}

	return VPNode, true
}

func (tree *VPTree) subTreeVectors(VPNode *VPNode) ([]Vector, error) {
	vectors := make([]Vector, 0)
	tree.inOrderTraversal(VPNode, &vectors)
	return vectors, nil
}

func (tree *VPTree) findMax(VPNode *VPNode) (Vector, bool) {
	if VPNode == nil {
		return Vector{}, false
	}
	if VPNode.Right != nil {
		return tree.findMax(VPNode.Right)
	}
	return VPNode.VantagePoint, true
}

func (tree *VPTree) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := tree.Insert(vec); err != nil {
			return err
		}
	}
	return nil
}

func (tree *VPTree) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := tree.Delete(vec); err != nil {
			return err
		}
	}
	return nil
}

func (tree *VPTree) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	var results []Vector
	tree.rangeSearchRecursive(tree.Root, query, radius, &results)
	return results, nil
}

func (tree *VPTree) rangeSearchRecursive(node *VPNode, query Vector, radius float64, results *[]Vector) {
	if node == nil {
		return
	}

	d := basic.EuclidDistanceVec(query, node.VantagePoint)

	if d <= radius {
		*results = append(*results, node.VantagePoint)
	}

	if d-radius < node.Mu {
		tree.rangeSearchRecursive(node.Left, query, radius, results)
	}
	if d+radius >= node.Mu {
		tree.rangeSearchRecursive(node.Right, query, radius, results)
	}
}

func (tree *VPTree) SaveToFile(filename string) error {
	// Note: This is a simple serialization implementation using encoding/gob.
	// Depending on the exact requirements, you might want a different serialization mechanism.
	dataFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer dataFile.Close()

	dataEncoder := gob.NewEncoder(dataFile)
	err = dataEncoder.Encode(tree)
	if err != nil {
		return err
	}
	return nil
}

func (tree *VPTree) LoadFromFile(filename string) error {
	dataFile, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer dataFile.Close()

	dataDecoder := gob.NewDecoder(dataFile)
	err = dataDecoder.Decode(tree)
	if err != nil {
		return err
	}
	return nil
}
