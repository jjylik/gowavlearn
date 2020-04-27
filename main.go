package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"

	"github.com/go-audio/wav"
)

const chunks = 10

type wavstats struct {
	AveragePitch []string
	Length       int64
	Rms          []string
}

var resultFile string
var game string
var imageDir string

func init() {
	flag.StringVar(&resultFile, "resultfile", "", "Classify data set source")
	flag.Parse()
}

func main() {
	log.Println(resultFile)
	if resultFile != "" {
		treeClassify(resultFile)
	} else {
		extractAllWavFeaturesToCsv()
	}

}
func extractAllWavFeaturesToCsv() {
	finished := make(chan bool)
	resultChan := make(chan []string, 10)
	numberOfFilesRead := 0
	numberOfFilesRead += readFilesInFolder("./data/kick", "Kick", resultChan)
	numberOfFilesRead += readFilesInFolder("./data/hat", "Hat", resultChan)
	numberOfFilesRead += readFilesInFolder("./data/other", "Other", resultChan)
	numberOfFilesRead += readFilesInFolder("./data/percussion", "Percussion", resultChan)
	numberOfFilesRead += readFilesInFolder("./data/snare", "Kick", resultChan)
	go writeResultCsv(resultChan, finished, numberOfFilesRead)
	<-finished
}

func readFilesInFolder(path, instrument string, resultChan chan []string) int {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		log.Fatal(err)
	}
	pattern := regexp.MustCompile(`.*\.wav`)
	expecting := 0
	for _, file := range files {
		if pattern.MatchString(file.Name()) {
			expecting++
			go processWav(filepath.Join(path, file.Name()), instrument, resultChan)
		}
	}
	return expecting
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

func writeResultCsv(resultChannel chan []string, finished chan<- bool, expecting int) {
	output, _ := os.Create("result.csv")
	writer := csv.NewWriter(output)
	writer.Write(createHeader())
	i := 0
	for line := range resultChannel {
		writer.Write(line)
		if i >= expecting {
			close(resultChannel)
		}
		i++
	}
	writer.Flush()
	output.Close()
	finished <- true
}

func processWav(file, instrument string, resultChannel chan<- []string) {
	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	d := wav.NewDecoder(f)
	stats, err := extractFeatures(d)
	if err != nil {
		return
	}
	line := []string{fmt.Sprintf("%d", stats.Length)}
	line = append(line, stats.AveragePitch...)
	line = append(line, stats.Rms...)
	line = append(line, instrument)
	resultChannel <- line
	log.Println("Processed ", file)
}

func normalize(val, min, max int64) float32 {
	return float32(val-min)/float32(max-min)*(1+1) - 1
}

func extractFeatures(d *wav.Decoder) (stats wavstats, err error) {
	length, _ := d.Duration()
	chunkLength := length / chunks
	d.Seek(0, 0)
	fullWavBuffer, err := d.FullPCMBuffer()
	sampleRate := int(d.Format().SampleRate)
	chunkBufferLength := int(chunkLength.Seconds() * float64(sampleRate))
	chunkBuffer := make([]float32, chunkBufferLength)
	stats.Length = length.Milliseconds()
	j := 0
	if d.NumChans > 2 {
		return stats, fmt.Errorf("too many chans")
	}
	for i := 0; i < len(fullWavBuffer.Data); i += int(d.NumChans) {
		var pcmValue int
		if d.NumChans == 2 {
			pcmValue = (fullWavBuffer.Data[i] + fullWavBuffer.Data[i+1]) / 2
		} else {
			pcmValue = fullWavBuffer.Data[i]
		}
		normalized := getNormalizedPcm(d.BitDepth, pcmValue)
		chunkBuffer[j] = normalized
		if j == chunkBufferLength-1 {
			frequency, _ := findMainFrequency(chunkBuffer, chunkBufferLength, sampleRate)
			rms := rootMeanSquare(chunkBuffer)
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

func getNormalizedPcm(bitDepth uint16, pcmValue int) float32 {
	if bitDepth == 16 {
		raw := int64(int32(int16(pcmValue)))
		return normalize(raw, -32768, 32767)
	} else if bitDepth == 24 {
		raw := int64(int32(pcmValue))
		return normalize(raw, -8388608, 8388607)
	} else if bitDepth == 32 {
		raw := int64(int32(pcmValue))
		return normalize(raw, -2147483648, 2147483647)
	}
	return 0
}

func rootMeanSquare(data []float32) float64 {
	sum := 0.
	n := float64(len(data))
	for _, x := range data {
		sum += float64(x * x)
	}
	return math.Sqrt(sum / n)
}
