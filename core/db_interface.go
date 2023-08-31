package core

import "hh_vectordb/basic"

type Vector = basic.Vector

// NearestNeighborSearch 基础的最近邻搜索
type NearestNeighborSearch interface {
	// Insert 插入
	Insert(vec Vector) error
	// Nearest 最近邻
	Nearest(query Vector) (Vector, error)

	KNearestSearch

	Vectors() ([]Vector, error)
	// Delete 删除
	Delete(vec Vector) error

	BatchOperator
	RangeSearch
	Persistence
}

// BatchOperator 批量操作
type BatchOperator interface {
	// InsertBatch 批量插入
	InsertBatch(vectors []Vector) error
	// DeleteBatch 批量删除
	DeleteBatch(vectors []Vector) error
}

// RangeSearch 范围搜索
type RangeSearch interface {
	SearchWithinRange(query Vector, radius float64) ([]Vector, error)
}

// KNearestSearch k近邻搜索
type KNearestSearch interface {
	KNearest(query Vector, k int) ([]Vector, error)
}

// Persistence 持久化
type Persistence interface {
	SaveToFile(filename string) error
	LoadFromFile(filename string) error
}

// Initializer 初始化
type Initializer interface {
	Init() error
}

// Cleanup 清理资源
type Cleanup interface {
	Close() error
}

// Statistics 统计、监控和优化
type Statistics interface {
	Stats() map[string]interface{}
}

// Concurrency 并发操作
type Concurrency interface {
	Lock()
	Unlock()
	RLock()   // 读锁
	RUnlock() // 释放读锁
}
