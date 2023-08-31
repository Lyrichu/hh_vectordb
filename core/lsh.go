package core

import (
	"encoding/gob"
	"errors"
	"hh_vectordb/basic"
	"math/rand"
	"os"
	"sort"
)

type LSH struct {
	HashTables    []map[int64][]Vector
	HashFuncs     []func(Vector) int64
	BucketSize    int
	RandomVectors []Vector
}

type lshGob struct {
	HashTables    []map[int64][]Vector
	BucketSize    int
	NumHashes     int
	RandomVectors []Vector
}

func NewLSH(numHashes int, bucketSize int) *LSH {
	hashFuncs := make([]func(Vector) int64, numHashes)
	hashTables := make([]map[int64][]Vector, numHashes)
	randomVectors := make([]Vector, numHashes)

	for i := range hashFuncs {
		hashFuncs[i], randomVectors[i] = randomHashFuncWithVector()
		hashTables[i] = make(map[int64][]Vector)
	}

	return &LSH{
		HashFuncs:     hashFuncs,
		HashTables:    hashTables,
		BucketSize:    bucketSize,
		RandomVectors: randomVectors,
	}
}

func (l *LSH) Insert(vec Vector) error {
	for i, hashFunc := range l.HashFuncs {
		hashValue := hashFunc(vec)
		bucket, exists := l.HashTables[i][hashValue]

		// If the bucket already has the maximum allowed vectors, don't insert the new vector.
		if exists && len(bucket) >= l.BucketSize {
			continue
		}

		if !exists {
			l.HashTables[i][hashValue] = []Vector{}
		}
		l.HashTables[i][hashValue] = append(l.HashTables[i][hashValue], vec)
	}
	return nil
}

func (l *LSH) Nearest(query Vector) (Vector, error) {
	candidates := l.getCandidates(query)

	var nearest Vector
	minDistance := float64(1 << 30) // some large number
	for _, vec := range candidates {
		if d := basic.EuclidDistanceVec(query, vec); d < minDistance {
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
		return basic.EuclidDistanceVec(query, candidates[i]) < basic.EuclidDistanceVec(query, candidates[j])
	})

	return candidates[:k], nil
}

func (l *LSH) Vectors() ([]Vector, error) {
	seen := make(map[int64]struct{}) // Use a map to keep track of seen vectors.
	var vectors []Vector
	for _, table := range l.HashTables {
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
	deletedFlag := false // This flag will be set to true if at least one instance of the vector is deleted

	for i, hashFunc := range l.HashFuncs {
		hashValue := hashFunc(vec)

		bucket, exists := l.HashTables[i][hashValue]
		if !exists {
			continue
		}

		// Create a new bucket excluding the vector to be deleted
		newBucket := make([]Vector, 0, len(bucket))
		for _, v := range bucket {
			if v.ID == vec.ID {
				deletedFlag = true
				continue // skip adding the deleted vector to newBucket
			}
			newBucket = append(newBucket, v)
		}

		// If newBucket is empty, delete the key from the map; otherwise, update the map with the new bucket
		if len(newBucket) == 0 {
			delete(l.HashTables[i], hashValue)
		} else {
			l.HashTables[i][hashValue] = newBucket
		}
	}

	if !deletedFlag {
		return errors.New("vector not found in any bucket")
	}

	return nil
}

func (l *LSH) getCandidates(query Vector) []Vector {
	seen := make(map[int64]bool)
	var candidates []Vector

	for i, hashFunc := range l.HashFuncs {
		hashValue := hashFunc(query)
		for _, vec := range l.HashTables[i][hashValue] {
			if !seen[vec.ID] {
				candidates = append(candidates, vec)
				seen[vec.ID] = true
			}
		}
	}

	return candidates
}

func (l *LSH) randomHashFunc() func(Vector) int64 {
	randomVec := randomVector()
	l.RandomVectors = append(l.RandomVectors, randomVec)
	return createHashFuncWithVector(randomVec)
}

func randomHashFuncWithVector() (func(Vector) int64, Vector) {
	randomVec := randomVector()
	return createHashFuncWithVector(randomVec), randomVec
}

func createHashFuncWithVector(vec Vector) func(Vector) int64 {
	return func(v Vector) int64 {
		return int64(basic.EuclidDistanceVec(vec, v))
	}
}

func randomVector() Vector {
	return Vector{
		Values: []float64{rand.Float64(), rand.Float64()},
	}
}

func (l *LSH) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := l.Insert(vec); err != nil {
			return err
		}
	}
	return nil
}

func (l *LSH) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		if err := l.Delete(vec); err != nil {
			return err
		}
	}
	return nil
}

func (l *LSH) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	candidates := l.getCandidates(query)
	var results []Vector
	for _, vec := range candidates {
		if d := basic.EuclidDistanceVec(query, vec); d <= radius {
			results = append(results, vec)
		}
	}
	if len(results) == 0 {
		return nil, errors.New("no vectors found within range")
	}
	return results, nil
}

func (l *LSH) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	aux := lshGob{
		HashTables:    l.HashTables,
		BucketSize:    l.BucketSize,
		RandomVectors: l.RandomVectors,
	}

	// Register types with gob. This ensures gob knows about our custom types and their nested structures.
	gob.Register(map[int64][]Vector{})
	gob.Register(Vector{})

	if err := encoder.Encode(&aux); err != nil {
		return err
	}
	return nil
}

func (l *LSH) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	aux := lshGob{}

	// Register types with gob. This ensures gob knows about our custom types and their nested structures.
	gob.Register(map[int64][]Vector{})
	gob.Register(Vector{})

	if err := decoder.Decode(&aux); err != nil {
		return err
	}

	l.HashTables = aux.HashTables
	l.BucketSize = aux.BucketSize
	l.RandomVectors = aux.RandomVectors

	l.HashFuncs = make([]func(Vector) int64, len(l.RandomVectors))
	for i, randomVec := range l.RandomVectors {
		l.HashFuncs[i] = createHashFuncWithVector(randomVec)
	}

	return nil
}
