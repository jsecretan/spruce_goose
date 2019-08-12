package main

import (
	"bufio"
	"fmt"
	"os"
	"time"
	"math"
	"math/rand"
	"encoding/binary"
	"strconv"
	"runtime"
	"io"
	"io/ioutil"
	"path/filepath"
	"sync"
	"github.com/aws/aws-sdk-go/aws"
  	"github.com/aws/aws-sdk-go/aws/awserr"
  	"github.com/aws/aws-sdk-go/aws/session"
  	"github.com/aws/aws-sdk-go/service/s3"
  	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"path"
	"bytes"
	"strings"
)

const CONNECTIONS_PER_NEURON = 1000
const NEURONS_IN_BLOCK int = 100000
const MAX_ITERATIONS int = 1
const NEURAL_DATA_BUCKET = "spruce-goose-neural-data"
// We consolidate value files, and here's how many we should put together
const FILES_PER_VALUE_BLOCK = 16
// We assume that workers > physical disks
const PHYSICAL_DISKS = 8

// Get blockfiles from the several physical disk directories
// Gets in an order that round-robins their physical disks
func getBlockFiles(baseDirectory string) []string {
	
	var diskGlobs [][]string
	totalLength := 0

	// Get the globs
	for disk := 0; disk < PHYSICAL_DISKS; disk++ {
		files, _ := filepath.Glob(path.Join(baseDirectory,fmt.Sprintf("neuronal%d",disk),"neural_block_*.nb"))
		diskGlobs = append(diskGlobs,files)
		totalLength += len(files)
	}

	blockFiles := make([]string, totalLength)

	for disk := 0; disk < PHYSICAL_DISKS; disk++ {
		//fmt.Printf("Get block files for disk %d with files %v\n", disk, diskGlobs[disk])
		for diskGlob := 0; diskGlob < len(diskGlobs[disk]); diskGlob++ {
			//fmt.Printf("Setting blockFiles[%d] = %s\n", disk+PHYSICAL_DISKS*diskGlob, diskGlobs[disk][diskGlob])
			blockFiles[disk+PHYSICAL_DISKS*diskGlob] = diskGlobs[disk][diskGlob]
		}
	}

	return blockFiles
}

// TODO we should interleave this too but not too worried about it now
func getValueFiles(baseDirectory string, currentIteration int) []string {

	var valueFilesToProcess []string

	for disk := 0; disk < PHYSICAL_DISKS; disk++ {
		outputFileGlob := path.Join(baseDirectory,fmt.Sprintf("neuronal%d",disk),strconv.Itoa(currentIteration),"neural_block_*.nb.output")
		valueFilesOnDisk, _ := filepath.Glob(outputFileGlob)
		valueFilesToProcess = append(valueFilesToProcess,valueFilesOnDisk...)
	}

	return valueFilesToProcess
}

// Translate an id so it rotates through the physical disks
func blockNumberToDirectoryPrefix(blockNumber int) string {
	return fmt.Sprintf("neuronal%d",blockNumber % PHYSICAL_DISKS)
}

// Writes a randomized neural connection file
func writeNeuralBlockWorker(totalNeurons uint64, neuralBlockFileChannel <-chan string, wg *sync.WaitGroup) {

	defer wg.Done()

	// Declare new rand generator because otherwise there are thread safety issues
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for fileName := range neuralBlockFileChannel {
		fmt.Printf("Writing %s \n", fileName)
		// Some kind of compression cheaper than gzip?
		file, _ := os.OpenFile(fileName, os.O_CREATE|os.O_WRONLY, 0600)
		defer file.Close()
		writer := bufio.NewWriter(file)

		rowBuffer := make([]byte, 8)
		connectionsPerNeuronBuffer := make([]byte, 4)
		connectionBuffer := make([]byte, 8)
		weightBuffer := make([]byte, 4)

		for row := 0; row < NEURONS_IN_BLOCK; row++ {
			binary.LittleEndian.PutUint64(rowBuffer, uint64(row))
			writer.Write(rowBuffer[0:5]) //Neuron number
			binary.LittleEndian.PutUint32(connectionsPerNeuronBuffer, uint32(CONNECTIONS_PER_NEURON))
			writer.Write(connectionsPerNeuronBuffer) // todo make a short and vary the number
			for col := 0; col < CONNECTIONS_PER_NEURON; col++ {
				connected_neuron := uint64(r.Int63n(int64(totalNeurons)))
				binary.LittleEndian.PutUint64(connectionBuffer, connected_neuron)
				// Trimming this to 5 bytes becuse that's the minimum we need to store 
				writer.Write(connectionBuffer[0:5])
				weight := r.Float32()
				binary.LittleEndian.PutUint32(weightBuffer, uint32(math.Float32bits(weight)))
	            writer.Write(weightBuffer)
			}
		}

		writer.Flush()
	}
}

// Load the latest iteration of weight files into in-memory array
func loadValueFilesToArray(neuronValuesCurrentStep []float32, iteration int) {
	//client, _ := hdfs.New("localhost:8020")
	directoryName := fmt.Sprintf("%d/values/", iteration)
	input := &s3.ListObjectsInput{
		Bucket:  aws.String(NEURAL_DATA_BUCKET),
		Prefix:	 aws.String(directoryName),
		MaxKeys: aws.Int64(1000), // TODO figure out what the limit should be
	}

	svc := s3.New(session.New())

	// The session the S3 Downloader will use
	sess := session.Must(session.NewSession())

	// Create a downloader with the session and default options
	downloader := s3manager.NewDownloader(sess)

	currentNeuron := 0

	// Example iterating over at most 3 pages of a ListObjects operation.
	svc.ListObjectsPages(input,
	    func(page *s3.ListObjectsOutput, lastPage bool) bool {

	    	fmt.Printf("Processing page of %d files\n", len(page.Contents))
			
			// TODO: Randomize?
			for _, objects := range page.Contents {

				buf := aws.NewWriteAtBuffer([]byte{})

				// Write the contents of S3 Object to the file
				n, _ := downloader.Download(buf, &s3.GetObjectInput{
				    Bucket: aws.String(NEURAL_DATA_BUCKET),
				    Key:    aws.String(*objects.Key),
				})

				fmt.Printf("file downloaded, %d bytes\n", n)

				weightData := buf.Bytes()

				for i := 0; i < len(weightData); i = i+4 {
					neuronValuesCurrentStep[currentNeuron] = math.Float32frombits(binary.LittleEndian.Uint32(weightData[i:i+4]))
					currentNeuron++
				}

			}

			return true
	    })

	fmt.Printf("Weights loaded = %d\n", currentNeuron)

}


// Block until the completion files show up in S3
func waitUntilOtherNodesDone(currentIteration int, totalNodes int) {

	svc := s3.New(session.New())

	for currentNode := 0; currentNode < totalNodes; currentNode++ {
		svc.WaitUntilObjectExists(&s3.HeadObjectInput{Bucket: aws.String(NEURAL_DATA_BUCKET), Key: aws.String(fmt.Sprintf("/%d/node%d.complete", currentIteration, currentNode))})
	}
}

// Put an object into S3 to signify that this node is done for this iteration
func markNodeAsDoneThisIteration(node int, currentIteration int) {

	svc := s3.New(session.New())

	input := &s3.PutObjectInput{
	   	Body: aws.ReadSeekCloser(strings.NewReader("1")),
	    Bucket: aws.String(NEURAL_DATA_BUCKET),
	    Key:    aws.String(fmt.Sprintf("/%d/node%d.complete", currentIteration, node)),
	}

	svc.PutObject(input)
}

// From https://play.golang.org/p/sR0vNRAQD1

type activation func(float64) float64

func sigmoid(x float64) float64 {
	return 1 / (1+math.Exp(x * (-1)))
}

func sigmoid_d(x float64) float64 {
	return math.Exp(x) / math.Pow((math.Exp(x) + 1), 2.0)
}

// Write out random values
func writeOutRandomValues(basePath string, node int, neuronStart uint64, neuronEnd uint64) {
	outputValueBuffer := make([]byte, 4)
	fileName := path.Join(basePath,fmt.Sprintf("neural_block_%d_0.nb.output", node))
	//fileName := fmt.Sprintf("neural_values_%d_0.nb.output", node)
	fmt.Println("Writing out random file\n")
	fmt.Println(fileName)
	outputfile, err := os.OpenFile(fileName, os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {
    	fmt.Errorf("failed to open file %q, %v", fileName, err)
	}

	defer outputfile.Close()

	writer := bufio.NewWriter(outputfile)

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for currentNeuron := neuronStart; currentNeuron < neuronEnd; currentNeuron++ {
		binary.LittleEndian.PutUint32(outputValueBuffer, uint32(math.Float32bits(r.Float32())))
    	writer.Write(outputValueBuffer)	
	}

	writer.Flush()
}

// Take a bunch of value files, and consolidate into fewer files up in HDFS
func consolidateFilesToDFS(iteration int, valueFileGroups <-chan []string, wg *sync.WaitGroup) {

	// TODO maybe I should use this uploader code:
// The session the S3 Uploader will use
	// sess := session.Must(session.NewSession())

	// // Create an uploader with the session and default options
	// uploader := s3manager.NewUploader(sess)

	// f, err  := os.Open(filename)
	// if err != nil {
	//     return fmt.Errorf("failed to open file %q, %v", filename, err)
	// }

	// // Upload the file to S3.
	// result, err := uploader.Upload(&s3manager.UploadInput{
	//     Bucket: aws.String(myBucket),
	//     Key:    aws.String(myString),
	//     Body:   f,
	// })
	// if err != nil {
	//     return fmt.Errorf("failed to upload file, %v", err)
	// }
	// fmt.Printf("file uploaded to, %s\n", aws.StringValue(result.Location))	


	defer wg.Done()

	svc := s3.New(session.New())

	for valueFileGroup := range valueFileGroups {

		fmt.Printf("Processing files %v\n", valueFileGroup)

		outputFileName := fmt.Sprintf("/%d/values/%s", iteration, path.Base(valueFileGroup[0]))

		var valueGroupBuffer []byte
		for _, valueFile := range valueFileGroup {
			//valueFileGroupFile, _ := os.OpenFile(valueFileGroup, os.O_RDONLY, 0644)
			b, _ := ioutil.ReadFile(valueFile)
			valueGroupBuffer = append(valueGroupBuffer,b...)
		}
		
		input := &s3.PutObjectInput{
		    //Body:   aws.ReadSeekCloser(bufio.NewReader(valueFileGroupFile)),
		   // Body:   aws.ReadSeekCloser(valueFileGroupFile),
		   	Body: bytes.NewReader(valueGroupBuffer),
		    Bucket: aws.String(NEURAL_DATA_BUCKET),
		    Key:    aws.String(outputFileName),
		}

		_, err := svc.PutObject(input)
		if err != nil {
		    if aerr, ok := err.(awserr.Error); ok {
		        switch aerr.Code() {
		        default:
		            fmt.Println(aerr.Error())
		        }
		    } else {
		        // Print the error, cast err to awserr.Error to get the Code and
		        // Message from an error.
		        fmt.Println(err.Error())
		    }
		    return
		}

	}
}

func neuralBlockWorker(neuronValuesCurrentStep []float32, iteration int, blockFiles <-chan string, fn activation, wg *sync.WaitGroup) {

	defer wg.Done()

	for fileName := range blockFiles {

		// Some kind of compression cheaper than gzip?
		blockfile, _ := os.OpenFile(fileName, os.O_RDONLY, 0644)
		defer blockfile.Close()

		// Open up file to store neuron output
		dir, blockFile := path.Split(fileName)
		outputFileName := path.Join(dir,strconv.Itoa(iteration),fmt.Sprintf("%s.output", blockFile))
		outputfile, _ := os.OpenFile(outputFileName, os.O_CREATE|os.O_WRONLY, 0644)
		defer outputfile.Close()

		reader := bufio.NewReader(blockfile)
		writer := bufio.NewWriter(outputfile)

		rowBuffer := make([]byte, 5)
		connectionsPerNeuronBuffer := make([]byte, 4)
		connectionBuffer := make([]byte, 5)
		weightBuffer := make([]byte, 4)
		outputValueBuffer := make([]byte, 4)

		for row := 0; row < NEURONS_IN_BLOCK; row++ {
			var sum float32 = 0.0
			io.ReadFull(reader, rowBuffer)
			// TODO should we do something with the neuron id
			//neuron_id := binary.LittleEndian.Uint64(append(rowBuffer,[]byte{0,0,0}...))
			//fmt.Printf("Neuron ID: %d\n", neuron_id)
			io.ReadFull(reader, connectionsPerNeuronBuffer)
			connectionsPerNeuron := int(binary.LittleEndian.Uint32(connectionsPerNeuronBuffer))
			//fmt.Printf("Iterating through %d connections\n", connectionsPerNeuron)
			for col := 0; col < connectionsPerNeuron; col++ {
				io.ReadFull(reader, connectionBuffer)
				connectedNeuron := binary.LittleEndian.Uint64(append(connectionBuffer,[]byte{0,0,0}...))
				io.ReadFull(reader, weightBuffer)
	            weight := math.Float32frombits(binary.LittleEndian.Uint32(weightBuffer))
	           	sum += weight*neuronValuesCurrentStep[connectedNeuron]	
	            //fmt.Printf("%d:%d:%f\n", col, connectedNeuron, weight)
			}

			binary.LittleEndian.PutUint32(outputValueBuffer, uint32(math.Float32bits(float32(sigmoid(float64(sum))))))
	        writer.Write(outputValueBuffer)

		}

		writer.Flush()

	}
}


func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	if len(os.Args) < 2 {
		fmt.Println("Missing node number")
		os.Exit(-1)
	}

	node,_ := strconv.Atoi(os.Args[1])
	baseDirectory := os.Args[3]
	var totalNodes int
	totalNodes,_ = strconv.Atoi(os.Args[2])
	totalNeuronsInt,_ := strconv.Atoi(os.Args[4])
	var totalNeurons = uint64(totalNeuronsInt)

	// Big ole' slice to store the neural values from the whole system
	neuronValuesCurrentStep := make([]float32, totalNeurons)
	
	var blocks_per_node uint64 = uint64(math.Ceil((float64(totalNeurons)/float64(NEURONS_IN_BLOCK))/float64(totalNodes)))
	// TODO Account for stragglers in the last node

	fmt.Printf("Running with %d blocks per node\n", blocks_per_node)

	// Worker per CPU
	numberOfWorkers := runtime.NumCPU()

	// The range of neurons this node owns
	var neuronStart uint64 = uint64(node) * blocks_per_node * uint64(NEURONS_IN_BLOCK)
	var neuronEnd uint64 = uint64(node + 1) * blocks_per_node * uint64(NEURONS_IN_BLOCK)

	time1 := time.Now()
	// Start by creating random sets of weights and connections, which we will read in and process
	// on subsequent interations
	neuralBlockFileChannel := make(chan string, blocks_per_node)
	for block := 0; block < int(blocks_per_node); block++ {
		neuralBlockFileChannel <- path.Join(baseDirectory,blockNumberToDirectoryPrefix(block),fmt.Sprintf("neural_block_%03d_%07d.nb", node, block))
	}
	close(neuralBlockFileChannel)

	// Make workers and write out those files
	var neuralBlockWg sync.WaitGroup
	neuralBlockWg.Add(numberOfWorkers)
	for w := 0; w < numberOfWorkers; w++ {
		go writeNeuralBlockWorker(totalNeurons, neuralBlockFileChannel, &neuralBlockWg)
	}
	neuralBlockWg.Wait()
	time2 := time.Now()
	fmt.Printf("Writing out neural block files took %v seconds\n", time2.Sub(time1))

	// Keep track of iterations of the network by number, main execution loop
	for currentIteration := 0; currentIteration <= MAX_ITERATIONS; currentIteration++ {

		fmt.Printf("Running iteration %d\n\n", currentIteration)
		// TODO how often should we report on block files, there can be hundreds of thousands per node
		//fmt.Printf("Running block file\n\n")

		// Make sure there is a scratch directory for iteration
		for disk := 0; disk < PHYSICAL_DISKS; disk++ {
			os.Mkdir(path.Join(baseDirectory,fmt.Sprintf("neuronal%d",disk),strconv.Itoa(currentIteration)), 0777)	
		}

		// On iteration 0, randomly generate neural values, otherwise we need to compute
		if currentIteration == 0 {
			// Initialize the random values
			// This is iteration "0"
			fmt.Printf("Initializing neuron values from %d to %d\n", neuronStart, neuronEnd)
			// Arbitrarily choosing the first physical disk, TODO break this up more later
			writeOutRandomValues(path.Join(baseDirectory,"neuronal0","0"), node, neuronStart, neuronEnd)
		} else {
			// Fill channel with block files to process
			// These neural block files contain connections and weights to execute the network
			blockFilesToProcess := getBlockFiles(baseDirectory)
			blockFilesChannel := make(chan string, len(blockFilesToProcess))

			fmt.Printf("Processing %d files\n", len(blockFilesToProcess))
			time1 = time.Now()
			// TODO, probably a cleaner way to copy the whole array into the channel
			for _, blockFile := range blockFilesToProcess {
				blockFilesChannel <- blockFile
			}
			close(blockFilesChannel)

			// Process all the block files, executing the network and determining the values
			// for all the neurons this node owns
			var blockLoadWg sync.WaitGroup
			blockLoadWg.Add(numberOfWorkers)
			for w := 0; w < numberOfWorkers; w++ {
	        	go neuralBlockWorker(neuronValuesCurrentStep, currentIteration, blockFilesChannel, sigmoid, &blockLoadWg)
	    	}
			blockLoadWg.Wait()
			time2 = time.Now()
			fmt.Printf("Processing neural block files took %v seconds\n", time2.Sub(time1))
		}

		// Take the value files to consolidate and upload
		// The value files store the values of each neuron for this iteration
		valueFilesToProcess := getValueFiles(baseDirectory, currentIteration)
		var valueFileGroups [][]string 

		fmt.Printf("Processing %d files\n\n", len(valueFilesToProcess))

		// Batch them together so we don't have tons of little files
		for FILES_PER_VALUE_BLOCK < len(valueFilesToProcess) {
    		valueFilesToProcess, valueFileGroups = valueFilesToProcess[FILES_PER_VALUE_BLOCK:], append(valueFileGroups, valueFilesToProcess[0:FILES_PER_VALUE_BLOCK:FILES_PER_VALUE_BLOCK])
		}
		valueFileGroups = append(valueFileGroups, valueFilesToProcess)

		// Turn it into a channel
		valueFilesChannel := make(chan []string, len(valueFileGroups))
		for _, valueFile := range valueFileGroups {
			valueFilesChannel <- valueFile
		}
		close(valueFilesChannel)

		// Sync the files to distributed storage so all the other nodes can pick them up before next iteration
		fmt.Printf("Syncing files to S3\n")
		var fileSyncWG sync.WaitGroup
		fileSyncWG.Add(numberOfWorkers)
		for w := 0; w < numberOfWorkers; w++ {
        	go consolidateFilesToDFS(currentIteration, valueFilesChannel, &fileSyncWG)
    	}
		fileSyncWG.Wait()

		// Signal that files for this iteration are processed and ready in S3 for other nodes to download
		markNodeAsDoneThisIteration(node, currentIteration)

		// Wait until everybody as marked done
		fmt.Printf("Waiting for everybody to be done\n")
		waitUntilOtherNodesDone(currentIteration, totalNodes)

		// Now iterate through all the files on S3
		fmt.Printf("Loading value files for next iteration\n")
		loadValueFilesToArray(neuronValuesCurrentStep, currentIteration)

	}

}