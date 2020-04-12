package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/go-audio/wav"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/filters"
	"github.com/sjwhitworth/golearn/trees"
)

const wavFile = "./Recorder_Flute_SI.wav"

const chunks = 10

type wavstats struct {
	AveragePitch []string
	Length       int64
	Rms          []string
}

func main() {
	err := splitWav()
	if err != nil {
		fmt.Println(err)
	}
}

func createHeader() []string {
	var header = []string{"Length"}
	for i := 0; i < chunks; i++ {
		header = append(header, fmt.Sprintf("Pitch_%d", i))
	}
	for i := 0; i < chunks; i++ {
		header = append(header, fmt.Sprintf("RMS_%d", i))
	}
	header = append(header, "Type")
	return header
}

func splitWav() error {
	f, err := os.Open(wavFile)
	if err != nil {
		return err
	}
	defer f.Close()
	d := wav.NewDecoder(f)
	stats, _ := extractFeatures(d)
	output, err := os.Create("result.csv")
	writer := csv.NewWriter(output)
	defer writer.Flush()
	writer.Write(createHeader())
	//	for i := 0; i < 1; i++ {
	line := []string{fmt.Sprintf("%d", stats.Length)}
	line = append(line, stats.AveragePitch...)
	line = append(line, stats.Rms...)
	line = append(line, "Drums")
	writer.Write(line)
	//}
	defer output.Close()
	writer.Flush()
	return nil
}

func normalize(val, min, max int64) float32 {
	return float32(val-min)/float32(max-min)*(1+1) - 1
}

func extractFeatures(d *wav.Decoder) (stats wavstats, err error) {
	length, _ := d.Duration()
	chunkLength := length / chunks
	d.Seek(0, 0)
	chunkBufferLength := int(chunkLength.Seconds() * 44100)
	fullWavBuffer, err := d.FullPCMBuffer()
	chunkBuffer := make([]float32, chunkBufferLength)
	j := 0
	stats.Length = length.Milliseconds()
	for _, s := range fullWavBuffer.Data {
		var normalized float32
		if d.BitDepth == 16 {
			raw := int64(int32(int16(s)))
			normalized = normalize(raw, -32768, 32767)
			fmt.Println(normalized)
		} else if d.BitDepth == 24 {
			raw := int64(int32(s))
			normalized = normalize(raw, -8388608, 8388607)
		}
		chunkBuffer[j] = normalized
		if j == chunkBufferLength-1 {
			frequency, probability := findMainFrequency(chunkBuffer, chunkBufferLength)
			rms := rootMeanSquare(chunkBuffer)
			fmt.Println(rms)
			fmt.Printf("Main Frequency: %f - Probability: %f \n", frequency, probability)
			stats.AveragePitch = append(stats.AveragePitch, fmt.Sprintf("%f", frequency))
			stats.Rms = append(stats.Rms, fmt.Sprintf("%f", rms))
			chunkBuffer = make([]float32, chunkBufferLength)
			j = 0
		}
		j++
	}
	if err == nil {
		err = d.Err()
	}
	return stats, err
}

func rootMeanSquare(data []float32) float64 {
	sum := 0.
	n := float64(len(data))
	for _, x := range data {
		sum += float64(x * x)
	}
	return math.Sqrt(sum / n)
}

func treeClassify() {
	var tree base.Classifier

	rand.Seed(44111342)

	// Load in the iris dataset
	iris, err := base.ParseCSVToInstances("datasets/iris_headers.csv", true)
	if err != nil {
		panic(err)
	}

	// Discretise the iris dataset with Chi-Merge
	filt := filters.NewChiMergeFilter(iris, 0.999)
	for _, a := range base.NonClassFloatAttributes(iris) {
		filt.AddAttribute(a)
	}
	filt.Train()
	irisf := base.NewLazilyFilteredInstances(iris, filt)

	// Create a 60-40 training-test split
	trainData, testData := base.InstancesTrainTestSplit(irisf, 0.60)

	//
	// First up, use ID3
	//
	tree = trees.NewID3DecisionTree(0.6)
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err := tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain)")
	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.InformationGainRatioRuleGenerator))
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain ratio)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.GiniCoefficientRuleGenerator))
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (gini index generator)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))
	//
	// Next up, Random Trees
	//

	// Consider two randomly-chosen attributes
	tree = trees.NewRandomTree(2)
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println("RandomTree Performance")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	//
	// Finally, Random Forests
	//
	tree = ensemble.NewRandomForest(70, 3)
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println("RandomForest Performance")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

}
