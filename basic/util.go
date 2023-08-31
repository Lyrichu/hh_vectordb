package basic

func VectorExistsInSlice(vec Vector, slice []Vector) bool {
	for _, v := range slice {
		if vec.Equals(v) {
			return true
		}
	}
	return false
}
