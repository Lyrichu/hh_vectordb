package core

// 暴力搜索算法

import (
	"encoding/gob"
	"errors"
	"hh_vectordb/basic"
	"math"
	"os"
	"sort"
)

type BruteForceSearch struct {
	data []Vector
}

func NewBruteForceSearch(vectors []Vector) *BruteForceSearch {
	searcher := &BruteForceSearch{}
	for _, vec := range vectors {
		err := searcher.Insert(vec)
		if err != nil {
			return nil
		}
	}
	return searcher
}

// Insert
//
//	@Description: 暴力搜索插入
//	@receiver b
//	@param vec 插入向量
//	@return error
func (b *BruteForceSearch) Insert(vec Vector) error {
	b.data = append(b.data, vec)
	return nil
}

// Nearest
//
//	@Description: 暴力搜索求解最近邻
//	@receiver b
//	@param query
//	@return Vector
//	@return error
func (b *BruteForceSearch) Nearest(query Vector) (Vector, error) {
	var nearest Vector
	var minDist = math.MaxFloat64

	for _, vec := range b.data {
		dist := basic.EuclidDistanceVec(vec, query)
		if dist < minDist {
			minDist = dist
			nearest = vec
		}
	}

	if minDist == math.MaxFloat64 {
		return Vector{}, errors.New("no vectors in the database")
	}

	return nearest, nil
}

// KNearest
//
//	@Description: 暴力搜索求解k-近邻
//	@receiver b
//	@param query
//	@param k
//	@return []Vector
//	@return error
func (b *BruteForceSearch) KNearest(query Vector, k int) ([]Vector, error) {
	type IDDist struct {
		Vector   Vector
		Distance float64
	}

	dists := make([]IDDist, len(b.data))
	for i, vec := range b.data {
		dists[i] = IDDist{
			Vector:   vec,
			Distance: basic.EuclidDistanceVec(query, vec),
		}
	}

	sort.Slice(dists, func(i, j int) bool {
		return dists[i].Distance < dists[j].Distance
	})

	if k > len(b.data) {
		k = len(b.data)
	}

	kNearest := make([]Vector, k)
	for i := 0; i < k; i++ {
		kNearest[i] = dists[i].Vector
	}

	return kNearest, nil
}

// Vectors
//
//	@Description:
//	@receiver b
//	@return []Vector
//	@return error
func (b *BruteForceSearch) Vectors() ([]Vector, error) {
	return b.data, nil
}

// Delete
//
//	@Description: 暴力搜索 删除向量
//	@receiver b
//	@param vec
//	@return error
func (b *BruteForceSearch) Delete(vec Vector) error {
	index := -1
	for i, v := range b.data {
		if v.Equals(vec) {
			index = i
			break
		}
	}

	if index == -1 {
		return errors.New("vector not found")
	}

	// 从切片中删除向量
	b.data = append(b.data[:index], b.data[index+1:]...)
	return nil
}

// InsertBatch implements the BatchOperator interface
//
//	@Description: 批量插入向量
//	@receiver b
//	@param vectors []Vector
//	@return error
func (b *BruteForceSearch) InsertBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := b.Insert(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

// DeleteBatch implements the BatchOperator interface
//
//	@Description: 批量删除向量
//	@receiver b
//	@param vectors []Vector
//	@return error
func (b *BruteForceSearch) DeleteBatch(vectors []Vector) error {
	for _, vec := range vectors {
		err := b.Delete(vec)
		if err != nil {
			return err
		}
	}
	return nil
}

// SearchWithinRange implements the RangeSearch interface for BruteForceSearch.
//
// @Description: Searches for vectors within a specified radius of the query vector.
// @receiver b
// @param query Vector - The query vector.
// @param radius float64 - The radius within which to search.
// @return []Vector - A slice of vectors within the specified radius.
// @return error - An error if something goes wrong.
func (b *BruteForceSearch) SearchWithinRange(query Vector, radius float64) ([]Vector, error) {
	var results []Vector

	for _, vec := range b.data {
		dist := basic.EuclidDistanceVec(vec, query)
		if dist <= radius {
			results = append(results, vec)
		}
	}

	if len(results) == 0 {
		return nil, errors.New("no vectors found within the specified range")
	}

	return results, nil
}

// SaveToFile implements the Persistence interface for BruteForceSearch.
//
// @Description: Saves the data slice to a file.
// @receiver b
// @param filename string - The name of the file to save to.
// @return error - An error if something goes wrong.
func (b *BruteForceSearch) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(b.data); err != nil {
		return err
	}

	return nil
}

// LoadFromFile implements the Persistence interface for BruteForceSearch.
//
// @Description: Loads the data slice from a file.
// @receiver b
// @param filename string - The name of the file to load from.
// @return error - An error if something goes wrong.
func (b *BruteForceSearch) LoadFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&b.data); err != nil {
		return err
	}

	return nil
}
