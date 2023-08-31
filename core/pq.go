package core

import (
	"container/heap"
	"encoding/gob"
	"errors"
	"hh_vectordb/basic"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

type Centroid struct {
	ID     int64
	Vector Vector
}

type PQ struct {
	m         int           // number of subvectors
	k         int           // number of centroids per subvector
	Codebooks [][]Centroid  // m x k Codebook
	DB        []Vector      // For simplicity, we'll also store the original vectors
	IDs       [][]int64     // Quantized IDs
	IDLookup  map[int64]int // Map from vector ID to its index in p.DB
}

// Compute an estimated distance for each encoded vector
type vectorDistPair struct {
	vector Vector
	dist   float64
}

type ChunkResult struct {
	Vectors []Vector
	Dists   []float64
}

type MaxHeap []vectorDistPair

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist } // Note the > for max heap
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(vectorDistPair))
}

func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func NewPQ(m, k int) *PQ {
	return &PQ{
		m:         m,
		k:         k,
		Codebooks: make([][]Centroid, m),
		IDLookup:  make(map[int64]int),
	}
}

func (p *PQ) Train(vectors []Vector, epochs int) {
	subvectorSize := len(vectors[0].Values) / p.m
	for i := 0; i < p.m; i++ {
		// Split vectors into subvectors for current group
		subvectors := make([]Vector, len(vectors))
		for j, vec := range vectors {
			subvectors[j] = Vector{
				Values: vec.Values[i*subvectorSize : (i+1)*subvectorSize],
			}
		}

		// Run k-means on subvectors
		centroids, _ := kmeans(subvectors, p.k, epochs, vectors)

		// Store the centroids in the codebook
		p.Codebooks[i] = centroids
	}
}

func kmeans(vectors []Vector, k, epochs int, originalVectors []Vector) ([]Centroid, error) {
	// 1. Initialize centroids randomly
	centroids := initializeCentroids(vectors, k)

	// 2. Iterate until convergence
	for iteration := 0; iteration < epochs; iteration++ { // let's set a max iteration count
		// Log current iteration
		if iteration%10 == 0 {
			log.Printf("K-means iteration: %d\n", iteration)
		}
		// Assign vectors to nearest centroids
		assignments := assignToNearest(vectors, centroids)

		// Compute new centroids
		newCentroids := computeCentroids(assignments, k, vectors)

		// Log centroids for this iteration
		for i, centroid := range newCentroids {
			log.Printf("Centroid %d: %v\n", i, centroid.Vector.Values)
		}

		// Check convergence (for simplicity, we'll check if centroids haven't changed)
		if centroidsEqual(centroids, newCentroids) {
			log.Println("Centroids converged!")
			break
		}

		centroids = newCentroids
	}

	return centroids, nil
}

func computeCentroids(assignments map[int][]Vector, k int, vectors []Vector) []Centroid {
	newCentroids := make([]Centroid, k)
	for idx, assignedVectors := range assignments {
		if len(assignedVectors) == 0 {
			// Re-initialize the centroid if no vectors are assigned to it
			randomIndex := rand.Intn(len(vectors))
			newCentroids[idx] = Centroid{ID: int64(idx), Vector: vectors[randomIndex]}
			continue
		}
		sum := make([]float64, len(assignedVectors[0].Values))
		for _, vec := range assignedVectors {
			for i, val := range vec.Values {
				sum[i] += val
			}
		}
		for i := range sum {
			sum[i] /= float64(len(assignedVectors))
		}
		newCentroids[idx] = Centroid{ID: int64(idx), Vector: Vector{Values: sum}}
	}
	return newCentroids
}

func initializeCentroids(vectors []Vector, k int) []Centroid {
	// Initialize the random seed
	rand.Seed(time.Now().UnixNano())

	// Shuffle the list of vectors
	rand.Shuffle(len(vectors), func(i, j int) {
		vectors[i], vectors[j] = vectors[j], vectors[i]
	})

	// Select the first 'k' vectors
	centroids := make([]Centroid, k)
	for i := 0; i < k; i++ {
		centroids[i] = Centroid{ID: vectors[i].ID, Vector: vectors[i]}
	}

	return centroids
}

func assignToNearest(vectors []Vector, centroids []Centroid) map[int][]Vector {
	assignments := make(map[int][]Vector)
	for _, vec := range vectors {
		minDist := math.MaxFloat64
		minIdx := 0
		for idx, centroid := range centroids {
			dist := basic.EuclidDistanceVec(vec, centroid.Vector)
			if dist < minDist {
				minDist = dist
				minIdx = idx
			}
		}
		assignments[minIdx] = append(assignments[minIdx], vec)
	}
	return assignments
}

func centroidsEqual(a, b []Centroid) bool {
	for i := range a {
		for j := range a[i].Vector.Values {
			if a[i].Vector.Values[j] != b[i].Vector.Values[j] {
				return false
			}
		}
	}
	return true
}

func (p *PQ) Insert(vec Vector) error {
	p.IDLookup[vec.ID] = len(p.DB) // Add to IDLookup
	p.DB = append(p.DB, vec)
	ids := p.quantize(vec)
	p.IDs = append(p.IDs, ids)
	return nil
}

func (p *PQ) quantize(vec Vector) []int64 {
	ids := make([]int64, p.m)
	subvectorSize := len(vec.Values) / p.m

	for i := 0; i < p.m; i++ {
		subvec := Vector{
			Values: vec.Values[i*subvectorSize : (i+1)*subvectorSize],
		}
		ids[i], _ = p.nearestCentroid(subvec, int64(i))
	}

	return ids
}

func (p *PQ) nearestCentroid(query Vector, mIndex int64) (int64, error) {
	minDist := math.MaxFloat64
	minIdx := int64(-1)
	for _, centroid := range p.Codebooks[mIndex] {
		dist := basic.EuclidDistanceVec(query, centroid.Vector)
		if dist < minDist {
			minDist = dist
			minIdx = centroid.ID
		}
	}

	if minIdx == -1 {
		return 0, errors.New("couldn't find a nearest centroid")
	}
	return minIdx, nil
}

func (p *PQ) Nearest(query Vector) (Vector, error) {
	if len(p.Codebooks) == 0 {
		return Vector{}, errors.New("codebook is not trained")
	}

	// Split the query into m segments
	segmentLength := len(query.Values) / p.m
	segments := splitVector(query.Values, segmentLength)

	// Calculate the distances from the query vector segments to all centroids
	distancesToCentroids := make([][]float64, p.m)
	for i, segment := range segments {
		distancesToCentroids[i] = p.calculateDistancesToCentroids(segment, p.Codebooks[i])
	}

	// Compute an estimated distance for each encoded vector and find the one with the smallest distance
	minDistance := math.MaxFloat64
	var closestVector Vector
	for _, vec := range p.DB {
		estimatedDist := p.estimateDistance(vec, distancesToCentroids)
		if estimatedDist < minDistance {
			minDistance = estimatedDist
			closestVector = vec
		}
	}

	return closestVector, nil
}

func (p *PQ) calculateDistancesToCentroids(segment []float64, centroids []Centroid) []float64 {
	var distances []float64
	for _, centroid := range centroids {
		dist := basic.EuclidDistance(segment, centroid.Vector.Values)
		distances = append(distances, dist)
	}
	return distances
}

func (p *PQ) estimateDistance(vec Vector, distancesToCentroids [][]float64) float64 {
	encodedParts := p.encodeVector(vec)

	totalDist := 0.0
	for i, part := range encodedParts {
		totalDist += distancesToCentroids[i][part]
	}
	return totalDist
}

func (p *PQ) encodeVector(vec Vector) []int64 {
	// Use the IDLookup map for faster index retrieval
	vecIndex, exists := p.IDLookup[vec.ID]
	if !exists {
		return nil
	}
	return p.IDs[vecIndex]
}

func (p *PQ) findClosestCentroid(segment []float64, centroids []Centroid) Centroid {
	var minDist = math.MaxFloat64
	var closestCentroid Centroid

	for _, centroid := range centroids {
		dist := basic.EuclidDistance(segment, centroid.Vector.Values)
		if dist < minDist {
			minDist = dist
			closestCentroid = centroid
		}
	}

	return closestCentroid
}

func (p *PQ) reconstructVector(centroids []Centroid) []float64 {
	var reconstructed []float64
	for _, centroid := range centroids {
		reconstructed = append(reconstructed, centroid.Vector.Values...)
	}
	return reconstructed
}

func splitVector(values []float64, segmentLength int) [][]float64 {
	var segments [][]float64
	for i := 0; i < len(values); i += segmentLength {
		end := i + segmentLength
		if end > len(values) {
			end = len(values)
		}
		segments = append(segments, values[i:end])
	}
	return segments
}

func (p *PQ) KNearest(query Vector, k int) ([]Vector, error) {
	if len(p.Codebooks) == 0 {
		return nil, errors.New("codebook is not trained")
	}

	// Split the query into m segments
	segmentLength := len(query.Values) / p.m
	segments := splitVector(query.Values, segmentLength)

	// Calculate the distances from the query vector segments to all centroids
	distancesToCentroids := make([][]float64, p.m)
	for i, segment := range segments {
		distancesToCentroids[i] = p.calculateDistancesToCentroids(segment, p.Codebooks[i])
	}
	h := &MaxHeap{}
	heap.Init(h)

	for _, vec := range p.DB {
		estimatedDist := p.estimateDistance(vec, distancesToCentroids)
		if h.Len() < k {
			heap.Push(h, vectorDistPair{vec, estimatedDist})
		} else if top := (*h)[0]; estimatedDist < top.dist {
			heap.Pop(h)
			heap.Push(h, vectorDistPair{vec, estimatedDist})
		}
	}

	// Extract top-k vectors from the heap
	result := make([]Vector, h.Len())
	for i := 0; i < len(result); i++ {
		pair := heap.Pop(h).(vectorDistPair)
		result[len(result)-1-i] = pair.vector
	}
	return result, nil
}

func (p *PQ) KNearestRefined(query Vector, k int) ([]Vector, error) {
	// Get a larger set of candidates using PQ
	candidateCount := k * 3 // Here we're using 5 times k, but you can adjust this multiplier
	candidates, err := p.KNearest(query, candidateCount)
	if err != nil {
		return nil, err
	}

	// Use max-heap to keep track of top-k vectors
	h := &MaxHeap{}
	heap.Init(h)

	for _, vec := range candidates {
		dist := basic.EuclidDistance(query.Values, vec.Values)
		if h.Len() < k {
			heap.Push(h, vectorDistPair{vec, dist})
		} else if top := (*h)[0]; dist < top.dist {
			heap.Pop(h)
			heap.Push(h, vectorDistPair{vec, dist})
		}
	}

	// Extract the results from the heap
	result := make([]Vector, h.Len())
	for i := 0; i < len(result); i++ {
		pair := heap.Pop(h).(vectorDistPair)
		result[len(result)-1-i] = pair.vector
	}
	return result, nil
}

func (p *PQ) Vectors() ([]Vector, error) {
	return p.DB, nil
}

func (p *PQ) Delete(vec Vector) error {
	indexToDelete, exists := p.IDLookup[vec.ID]
	if !exists {
		return errors.New("vector not found in the database")
	}
	// Remove vector from p.DB and update IDLookup map
	p.DB = append(p.DB[:indexToDelete], p.DB[indexToDelete+1:]...)
	delete(p.IDLookup, vec.ID)

	// Adjust IDLookup indices for vectors after the deleted vector
	for i := indexToDelete; i < len(p.DB); i++ {
		p.IDLookup[p.DB[i].ID] = i
	}
	// Remove IDs from p.IDs
	p.IDs = append(p.IDs[:indexToDelete], p.IDs[indexToDelete+1:]...)
	return nil
}

func (p *PQ) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := p.Insert(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *PQ) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := p.Delete(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *PQ) SearchWithinInterval(query Vector, minDist float64, maxDist float64) ([]Vector, error) {
	var result []Vector
	subVectorLength := len(query.Values) / p.m
	n := 3                           // consider the top 3 centroids, adjust based on your needs
	expandedMaxDist := 3.0 * maxDist // 20% expansion, adjust based on your needs

	candidateIndices := make(map[int]struct{})

	// Helper function to find the top n the closest centroids for a subvector.
	findTopNCentroids := func(subVec []float64, centroids []Centroid, n int) []Centroid {
		type DistCentroidPair struct {
			dist     float64
			centroid Centroid
		}
		pairs := make([]DistCentroidPair, len(centroids))
		for i, c := range centroids {
			pairs[i] = DistCentroidPair{
				dist:     basic.EuclidDistance(subVec, c.Vector.Values),
				centroid: c,
			}
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].dist < pairs[j].dist
		})

		topN := make([]Centroid, n)
		for i := 0; i < n && i < len(pairs); i++ {
			topN[i] = pairs[i].centroid
		}
		return topN
	}

	// 1. Quantize the query to find top n centroids.
	for i := 0; i < p.m; i++ {
		start := i * subVectorLength
		end := start + subVectorLength
		subVec := query.Values[start:end]

		// find the top n centroids for this subvector
		topCentroids := findTopNCentroids(subVec, p.Codebooks[i], n)

		for _, centroid := range topCentroids {
			distToCentroid := basic.EuclidDistance(subVec, centroid.Vector.Values)
			if distToCentroid <= expandedMaxDist {
				for idx, _ := range p.IDs[i] {
					candidateIndices[idx] = struct{}{}
				}
			}
		}
	}

	// 2. Refinement step
	for idx := range candidateIndices {
		dist := basic.EuclidDistance(query.Values, p.DB[idx].Values)
		if dist >= minDist && dist <= maxDist {
			result = append(result, p.DB[idx])
		}
	}

	return result, nil
}

func (p *PQ) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	return p.SearchWithinInterval(query, 0, radius)
}

func (p *PQ) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(p); err != nil {
		return err
	}
	return nil
}

func (p *PQ) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(p); err != nil {
		return err
	}

	return nil
}

func (p *PQ) processChunk(chunk []Vector, distancesToCentroids [][]float64, ch chan<- ChunkResult) {
	vectors := make([]Vector, 0)
	dists := make([]float64, 0)

	for _, vec := range chunk {
		estimatedDist := p.estimateDistance(vec, distancesToCentroids)
		vectors = append(vectors, vec)
		dists = append(dists, estimatedDist)
	}

	ch <- ChunkResult{vectors, dists}
}

func (p *PQ) KNearestConcurrent(query Vector, k int) ([]Vector, error) {

	if len(p.Codebooks) == 0 {
		return nil, errors.New("codebook is not trained")
	}

	// Split the query into m segments
	segmentLength := len(query.Values) / p.m
	segments := splitVector(query.Values, segmentLength)

	// Calculate the distances from the query vector segments to all centroids
	distancesToCentroids := make([][]float64, p.m)
	for i, segment := range segments {
		distancesToCentroids[i] = p.calculateDistancesToCentroids(segment, p.Codebooks[i])
	}

	numCores := runtime.NumCPU()
	chunkSize := len(p.DB) / numCores
	ch := make(chan ChunkResult, numCores)

	// Split the DB and start the goroutines
	for i := 0; i < numCores; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == numCores-1 {
			end = len(p.DB)
		}
		go p.processChunk(p.DB[start:end], distancesToCentroids, ch)
	}

	h := &MaxHeap{}
	heap.Init(h)
	mu := &sync.Mutex{} // Mutex for thread-safe operations on the heap

	// Collect results from goroutines
	for i := 0; i < numCores; i++ {
		result := <-ch
		for j, vec := range result.Vectors {
			estimatedDist := result.Dists[j]
			mu.Lock()
			if h.Len() < k {
				heap.Push(h, vectorDistPair{vec, estimatedDist})
			} else if top := (*h)[0]; estimatedDist < top.dist {
				heap.Pop(h)
				heap.Push(h, vectorDistPair{vec, estimatedDist})
			}
			mu.Unlock()
		}
	}

	// Extract top-k vectors from the heap
	result := make([]Vector, h.Len())
	for i := 0; i < len(result); i++ {
		pair := heap.Pop(h).(vectorDistPair)
		result[len(result)-1-i] = pair.vector
	}

	return result, nil
}
