package core

import (
	"encoding/gob"
	"errors"
	"hh_vectordb/basic"
	"math"
	"os"
	"sort"
)

type CoverTreeNode struct {
	Point     Vector
	Level     int
	Children  []*CoverTreeNode
	MaxMetric float64
}

type CoverTree struct {
	Root *CoverTreeNode
	Size int
	Base float64
}

func NewCoverTree(base float64) *CoverTree {
	return &CoverTree{Base: base}
}

func (ct *CoverTree) Insert(vec Vector) error {
	if ct.Root == nil {
		ct.Root = &CoverTreeNode{Point: vec, Level: 0}
		return nil
	}

	err := ct.insert(ct.Root, vec)
	if err == nil {
		return nil
	}

	if err.Error() != "cannot insert" {
		return err
	}

	// Only create a new root if there's no other option
	newRoot := &CoverTreeNode{
		Point:    vec,
		Level:    ct.Root.Level + 1,
		Children: []*CoverTreeNode{ct.Root},
	}
	ct.Root = newRoot
	return nil
}

func (ct *CoverTree) insert(node *CoverTreeNode, vec Vector) error {
	d := basic.EuclidDistanceVec(node.Point, vec)
	if d == 0 {
		return errors.New("duplicate vector")
	}

	childLevel := node.Level - 1
	if d < math.Pow(ct.Base, float64(childLevel)) {
		for _, child := range node.Children {
			if err := ct.insert(child, vec); err == nil {
				return nil
			}
		}
		child := &CoverTreeNode{Point: vec, Level: childLevel}
		node.Children = append(node.Children, child)
		return nil
	}
	return errors.New("cannot insert")
}

func (ct *CoverTree) Nearest(query Vector) (Vector, error) {
	_, vec, err := ct.nearest(ct.Root, query, math.MaxFloat64)
	return vec, err
}

func (ct *CoverTree) nearest(node *CoverTreeNode, query Vector, currentBest float64) (float64, Vector, error) {
	if node == nil {
		return currentBest, Vector{}, nil
	}
	d := basic.EuclidDistanceVec(node.Point, query)
	if d < currentBest {
		currentBest = d
	}

	bestDist := currentBest
	bestVec := node.Point

	for _, child := range node.Children {
		if basic.EuclidDistanceVec(child.Point, query)-math.Pow(ct.Base, float64(child.Level)) < currentBest {
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

	d := basic.EuclidDistanceVec(node.Point, query)
	if d < currentBestDistance {
		currentBestDistance = d
	}
	bestDist := d
	bestVec := node.Point

	for _, child := range node.Children {
		// Pruning step: Compute the minimum distance from the query to any point in child's subtree
		// Note: This is a simplistic bound. You can use more sophisticated bounds based on Cover Tree properties
		bound := basic.EuclidDistanceVec(child.Point, query) - math.Pow(ct.Base, float64(child.Level))

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
	if ct.Root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	results := make([]Vector, 0, k)
	ct.kNearest(ct.Root, query, &results, k)
	return results, nil
}

func (ct *CoverTree) kNearest(node *CoverTreeNode, query Vector, results *[]Vector, k int) {
	if node == nil {
		return
	}

	d := basic.EuclidDistanceVec(node.Point, query)

	// Check if this node's point should be in the top-k results
	if len(*results) < k {
		*results = append(*results, node.Point)
	} else {
		maxDist := basic.EuclidDistanceVec((*results)[k-1], query)
		if d < maxDist {
			(*results)[k-1] = node.Point
		}
	}

	// Sort results by distance to ensure only top-k are kept
	sort.Slice(*results, func(i, j int) bool {
		return basic.EuclidDistanceVec((*results)[i], query) < basic.EuclidDistanceVec((*results)[j], query)
	})

	// Recurse into children nodes
	for _, child := range node.Children {
		ct.kNearest(child, query, results, k)
	}
}

func (ct *CoverTree) Vectors() ([]Vector, error) {
	if ct.Root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	var results []Vector
	ct.collectVectors(ct.Root, &results)
	return results, nil
}

func (ct *CoverTree) collectVectors(node *CoverTreeNode, results *[]Vector) {
	if node == nil {
		return
	}

	*results = append(*results, node.Point)
	for _, child := range node.Children {
		ct.collectVectors(child, results)
	}
}

func (ct *CoverTree) Delete(vec Vector) error {
	if ct.Root == nil {
		return errors.New("tree is empty")
	}

	if ct.Root.Point.Equals(vec) {
		if len(ct.Root.Children) == 0 {
			ct.Root = nil
		} else {
			newRoot := ct.Root.Children[0]
			remainingChildren := ct.Root.Children[1:]
			ct.Root = newRoot
			for _, child := range remainingChildren {
				err := ct.insert(ct.Root, child.Point)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}

	return ct.delete(ct.Root, vec)
}

func (ct *CoverTree) delete(node *CoverTreeNode, vec Vector) error {
	for i, child := range node.Children {
		if child.Point.Equals(vec) {
			// if the node to be deleted has children, promote one of them
			if len(child.Children) > 0 {
				promote := child.Children[0]
				remainingChildren := child.Children[1:]
				node.Children[i] = promote
				for _, grandChild := range remainingChildren {
					err := ct.insert(node, grandChild.Point)
					if err != nil {
						return err
					} // Insert remaining children at current node level
				}
			} else {
				// remove the child from the children slice
				node.Children = append(node.Children[:i], node.Children[i+1:]...)
			}
			return nil
		}
	}

	for _, child := range node.Children {
		err := ct.delete(child, vec)
		if err == nil {
			return nil
		}
	}

	return errors.New("vector not found in the tree")
}

func (ct *CoverTree) KNearestV2(query Vector, k int) ([]Vector, error) {
	if ct.Root == nil {
		return []Vector{}, errors.New("tree is empty")
	}

	// We'll maintain a list of vectors (results) and distances (currentBest).
	results := make([]Vector, 0, k)
	currentBest := make([]float64, 0, k)

	ct.kNearestV2(ct.Root, query, &results, &currentBest, k)
	return results, nil
}

func (ct *CoverTree) kNearestV2(node *CoverTreeNode, query Vector, results *[]Vector, currentBest *[]float64, k int) {
	if node == nil {
		return
	}

	d := basic.EuclidDistanceVec(node.Point, query)

	if len(*results) < k {
		*results = append(*results, node.Point)
		*currentBest = append(*currentBest, d)
		sortLastAdded(results, currentBest, query)
	} else {
		maxDist := (*currentBest)[k-1]
		if d < maxDist {
			(*results)[k-1] = node.Point
			(*currentBest)[k-1] = d
			sortLastAdded(results, currentBest, query)
		}
	}

	// Pruning step
	if len(*currentBest) == k {
		maxDist := (*currentBest)[k-1]
		bound := basic.EuclidDistanceVec(node.Point, query) - math.Pow(ct.Base, float64(node.Level))
		if bound >= maxDist {
			return
		}
	}

	// Recurse into children nodes
	for _, child := range node.Children {
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

func (ct *CoverTree) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := ct.Insert(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (ct *CoverTree) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := ct.Delete(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (ct *CoverTree) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	var results []Vector
	ct.searchWithinRange(ct.Root, query, radius, &results)
	return results, nil
}

func (ct *CoverTree) searchWithinRange(node *CoverTreeNode, query Vector, radius float64, results *[]Vector) {
	if node == nil {
		return
	}

	if basic.EuclidDistanceVec(node.Point, query) <= radius {
		*results = append(*results, node.Point)
	}

	for _, child := range node.Children {
		bound := basic.EuclidDistanceVec(child.Point, query) - math.Pow(ct.Base, float64(child.Level))
		if bound <= radius {
			ct.searchWithinRange(child, query, radius, results)
		}
	}
}

func (ct *CoverTree) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(ct)
	return err
}

func (ct *CoverTree) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(ct)
	return err
}
