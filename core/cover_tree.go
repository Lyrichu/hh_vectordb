package core

import (
	"errors"
	"math"
	"sort"
)

type CoverTreeNode struct {
	point     Vector
	level     int
	children  []*CoverTreeNode
	maxMetric float64
}

type CoverTree struct {
	root *CoverTreeNode
	size int
	base float64
}

func NewCoverTree(base float64) *CoverTree {
	return &CoverTree{base: base}
}

func distance(a, b Vector) float64 {
	sum := 0.0
	for i := range a.Values {
		d := a.Values[i] - b.Values[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func (ct *CoverTree) Insert(vec Vector) error {
	if ct.root == nil {
		ct.root = &CoverTreeNode{point: vec, level: 0}
		return nil
	}

	err := ct.insert(ct.root, vec)
	if err == nil {
		return nil
	}

	if err.Error() != "cannot insert" {
		return err
	}

	// Only create a new root if there's no other option
	newRoot := &CoverTreeNode{
		point:    vec,
		level:    ct.root.level + 1,
		children: []*CoverTreeNode{ct.root},
	}
	ct.root = newRoot
	return nil
}

func (ct *CoverTree) insert(node *CoverTreeNode, vec Vector) error {
	d := distance(node.point, vec)
	if d == 0 {
		return errors.New("duplicate vector")
	}

	childLevel := node.level - 1
	if d < math.Pow(ct.base, float64(childLevel)) {
		for _, child := range node.children {
			if err := ct.insert(child, vec); err == nil {
				return nil
			}
		}
		child := &CoverTreeNode{point: vec, level: childLevel}
		node.children = append(node.children, child)
		return nil
	}
	return errors.New("cannot insert")
}

func (ct *CoverTree) Nearest(query Vector) (Vector, error) {
	_, vec, err := ct.nearest(ct.root, query, math.MaxFloat64)
	return vec, err
}

func (ct *CoverTree) nearest(node *CoverTreeNode, query Vector, currentBest float64) (float64, Vector, error) {
	if node == nil {
		return currentBest, Vector{}, nil
	}
	d := distance(node.point, query)
	if d < currentBest {
		currentBest = d
	}

	bestDist := currentBest
	bestVec := node.point

	for _, child := range node.children {
		if distance(child.point, query)-math.Pow(ct.base, float64(child.level)) < currentBest {
			dist, vec, err := ct.nearest(child, query, bestDist)
			if err != nil {
				return bestDist, bestVec, err
			}
			if dist < bestDist {
				bestDist = dist
				bestVec = vec
			}
		}
	}
	return bestDist, bestVec, nil
}

func (ct *CoverTree) nearestV2(node *CoverTreeNode, query Vector, currentBestDistance float64) (float64, Vector, error) {
	if node == nil {
		return math.MaxFloat64, Vector{}, errors.New("node is nil")
	}

	d := distance(node.point, query)
	if d < currentBestDistance {
		currentBestDistance = d
	}
	bestDist := d
	bestVec := node.point

	for _, child := range node.children {
		// Pruning step: Compute the minimum distance from the query to any point in child's subtree
		// Note: This is a simplistic bound. You can use more sophisticated bounds based on Cover Tree properties
		bound := distance(child.point, query) - math.Pow(ct.base, float64(child.level))

		if bound > currentBestDistance {
			continue // Prune this branch
		}

		dist, vec, _ := ct.nearestV2(child, query, bestDist)
		if dist < bestDist {
			bestDist = dist
			bestVec = vec
		}
	}
	return bestDist, bestVec, nil
}

func (ct *CoverTree) KNearest(query Vector, k int) ([]Vector, error) {
	if ct.root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	results := make([]Vector, 0, k)
	ct.kNearest(ct.root, query, &results, k)
	return results, nil
}

func (ct *CoverTree) kNearest(node *CoverTreeNode, query Vector, results *[]Vector, k int) {
	if node == nil {
		return
	}

	d := distance(node.point, query)

	// Check if this node's point should be in the top-k results
	if len(*results) < k {
		*results = append(*results, node.point)
	} else {
		maxDist := distance((*results)[k-1], query)
		if d < maxDist {
			(*results)[k-1] = node.point
		}
	}

	// Sort results by distance to ensure only top-k are kept
	sort.Slice(*results, func(i, j int) bool {
		return distance((*results)[i], query) < distance((*results)[j], query)
	})

	// Recurse into children nodes
	for _, child := range node.children {
		ct.kNearest(child, query, results, k)
	}
}

func (ct *CoverTree) Vectors() ([]Vector, error) {
	if ct.root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	var results []Vector
	ct.collectVectors(ct.root, &results)
	return results, nil
}

func (ct *CoverTree) collectVectors(node *CoverTreeNode, results *[]Vector) {
	if node == nil {
		return
	}

	*results = append(*results, node.point)
	for _, child := range node.children {
		ct.collectVectors(child, results)
	}
}

func (ct *CoverTree) Delete(vec Vector) error {
	if ct.root == nil {
		return errors.New("tree is empty")
	}

	if ct.root.point.Equals(vec) {
		if len(ct.root.children) == 0 {
			ct.root = nil
		} else {
			ct.root = ct.root.children[0]
			for _, child := range ct.root.children[1:] {
				err := ct.Insert(child.point)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}

	return ct.delete(ct.root, vec)
}

func (ct *CoverTree) delete(node *CoverTreeNode, vec Vector) error {
	for i, child := range node.children {
		if child.point.Equals(vec) {
			// if the node to be deleted has children, promote one of them
			if len(child.children) > 0 {
				promote := child.children[0]
				node.children[i] = promote
				for _, grandChild := range child.children[1:] {
					err := ct.Insert(grandChild.point)
					if err != nil {
						return err
					}
				}
			} else {
				// remove the child from the children slice
				node.children = append(node.children[:i], node.children[i+1:]...)
			}
			return nil
		}
	}

	for _, child := range node.children {
		err := ct.delete(child, vec)
		if err == nil {
			return nil
		}
	}

	return errors.New("vector not found in the tree")
}

func (ct *CoverTree) KNearestV2(query Vector, k int) ([]Vector, error) {
	if ct.root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	// We'll maintain a list of vectors (results) and distances (currentBest).
	results := make([]Vector, 0, k)
	currentBest := make([]float64, 0, k)

	ct.kNearestV2(ct.root, query, &results, &currentBest, k)
	return results, nil
}

func (ct *CoverTree) kNearestV2(node *CoverTreeNode, query Vector, results *[]Vector, currentBest *[]float64, k int) {
	if node == nil {
		return
	}

	d := distance(node.point, query)

	if len(*results) < k {
		*results = append(*results, node.point)
		*currentBest = append(*currentBest, d)
		sortLastAdded(results, currentBest, query)
	} else {
		maxDist := (*currentBest)[k-1]
		if d < maxDist {
			(*results)[k-1] = node.point
			(*currentBest)[k-1] = d
			sortLastAdded(results, currentBest, query)
		}
	}

	// Pruning step
	if len(*currentBest) == k {
		maxDist := (*currentBest)[k-1]
		bound := distance(node.point, query) - math.Pow(ct.base, float64(node.level))
		if bound >= maxDist {
			return
		}
	}

	// Recurse into children nodes
	for _, child := range node.children {
		ct.kNearestV2(child, query, results, currentBest, k)
	}
}

// Utility function to sort only the last added element in results and currentBest
func sortLastAdded(results *[]Vector, currentBest *[]float64, query Vector) {
	i := len(*currentBest) - 1
	for i > 0 && (*currentBest)[i] < (*currentBest)[i-1] {
		(*currentBest)[i], (*currentBest)[i-1] = (*currentBest)[i-1], (*currentBest)[i]
		(*results)[i], (*results)[i-1] = (*results)[i-1], (*results)[i]
		i--
	}
}
