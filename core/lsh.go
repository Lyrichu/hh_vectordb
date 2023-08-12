package core

import (
	"errors"
	"hh_vectordb/basic"
	"math/rand"
	"sort"
)

type LSH struct {
	hashTables []map[int64][]Vector
	hashFuncs  []func(Vector) int64
	bucketSize int
}

func NewLSH(numHashes int, bucketSize int) *LSH {
	hashFuncs := make([]func(Vector) int64, numHashes)
	hashTables := make([]map[int64][]Vector, numHashes)

	for i := range hashFuncs {
		hashFuncs[i] = randomHashFunc()
		hashTables[i] = make(map[int64][]Vector)
	}

	return &LSH{
		hashFuncs:  hashFuncs,
		hashTables: hashTables,
		bucketSize: bucketSize,
	}
}

func (l *LSH) Insert(vec Vector) error {
	for i, hashFunc := range l.hashFuncs {
		hashValue := hashFunc(vec)
		bucket, exists := l.hashTables[i][hashValue]

		// If the bucket already has the maximum allowed vectors, don't insert the new vector.
		if exists && len(bucket) >= l.bucketSize {
			continue
		}

		if !exists {
			l.hashTables[i][hashValue] = []Vector{}
		}
		l.hashTables[i][hashValue] = append(l.hashTables[i][hashValue], vec)
	}
	return nil
}

func (l *LSH) Nearest(query Vector) (Vector, error) {
	candidates := l.getCandidates(query)

	var nearest Vector
	minDistance := float64(1 << 30) // some large number
	for _, vec := range candidates {
		if d := distance(query, vec); d < minDistance {
			nearest = vec
			minDistance = d
		}
	}

	if minDistance == float64(1<<30) {
		return Vector{}, errors.New("no neighbors found")
	}

	return nearest, nil
}

func (l *LSH) KNearest(query Vector, k int) ([]Vector, error) {
	candidates := l.getCandidates(query)

	if len(candidates) < k {
		return nil, errors.New("not enough neighbors found")
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		return distance(query, candidates[i]) < distance(query, candidates[j])
	})

	return candidates[:k], nil
}

func (l *LSH) Vectors() ([]Vector, error) {
	seen := make(map[int64]struct{}) // Use a map to keep track of seen vectors.
	var vectors []Vector
	for _, table := range l.hashTables {
		for _, bucket := range table {
			for _, vec := range bucket {
				if _, found := seen[vec.ID]; !found {
					vectors = append(vectors, vec)
					seen[vec.ID] = struct{}{}
				}
			}
		}
	}
	return vectors, nil
}

func (l *LSH) Delete(vec Vector) error {
	for i, hashFunc := range l.hashFuncs {
		hashValue := hashFunc(vec)
		if bucket, exists := l.hashTables[i][hashValue]; exists {
			newBucket := make([]Vector, 0, len(bucket))
			for _, v := range bucket {
				if v.ID != vec.ID {
					newBucket = append(newBucket, v)
				}
			}
			l.hashTables[i][hashValue] = newBucket
		}
	}
	return nil
}

func (l *LSH) getCandidates(query Vector) []Vector {
	seen := make(map[int64]bool)
	var candidates []Vector

	for i, hashFunc := range l.hashFuncs {
		hashValue := hashFunc(query)
		for _, vec := range l.hashTables[i][hashValue] {
			if !seen[vec.ID] {
				candidates = append(candidates, vec)
				seen[vec.ID] = true
			}
		}
	}

	return candidates
}

func randomHashFunc() func(Vector) int64 {
	// This is a very basic hash function example. LSH typically uses more complex family of hash functions.
	randomVec := randomVector()
	return func(vec Vector) int64 {
		return int64(basic.EuclidDistanceVec(randomVec, vec))
	}
}

func randomVector() Vector {
	return Vector{
		Values: []float64{rand.Float64(), rand.Float64()},
	}
}
