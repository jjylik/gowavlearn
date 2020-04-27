package main

import (
	"fmt"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/filters"
	"github.com/sjwhitworth/golearn/trees"
)

func treeClassify(csvFile string) {
	var tree base.Classifier

	rand.Seed(44111342)

	features, err := base.ParseCSVToInstances(csvFile, true)
	if err != nil {
		panic(err)
	}

	filt := filters.NewChiMergeFilter(features, 0.999)
	for _, a := range base.NonClassFloatAttributes(features) {
		filt.AddAttribute(a)
	}
	filt.Train()
	featuresFiltered := base.NewLazilyFilteredInstances(features, filt)

	trainData, testData := base.InstancesTrainTestSplit(featuresFiltered, 0.60)

	tree = trees.NewID3DecisionTree(0.6)

	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	predictions, err := tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.InformationGainRatioRuleGenerator))

	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	fmt.Println("ID3 Performance (information gain ratio)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.GiniCoefficientRuleGenerator))

	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println("ID3 Performance (gini index generator)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

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
