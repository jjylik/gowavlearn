package main

import "testing"

func TestNormalize(t *testing.T) {
	got := normalize(-2, -10, 10)
	if 0.2-got < 0.00 {
		t.Errorf("normalize(-2) = %f", got)
	}
}
