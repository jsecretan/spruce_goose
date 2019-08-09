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
	"github.com/go-redis/redis"
	"github.com/aws/aws-sdk-go/aws"
  	"github.com/aws/aws-sdk-go/aws/awserr"
  	"github.com/aws/aws-sdk-go/aws/session"
  	"github.com/aws/aws-sdk-go/service/s3"
  	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"errors"
	"path"
	"bytes"
)

const CONNECTIONS_PER_NEURON = 1000
const NEURONS_IN_BLOCK int = 10000
const TOTAL_NEURONS uint64 = 100000
const TOTAL_NODES uint64 = 1
const MAX_ITERATIONS int = 1
const NEURAL_DATA_BUCKET = "spruce-goose-neural-data"
const REDIS_SERVER = "localhost:6379"
const BASE_DIRECTORY = "/Users/jsecretan/code/spruce_goose"
// We consolidate value files, and here's how many we should put together
const FILES_PER_VALUE_BLOCK = 16
// We assume that workers > physical disks
const PHYSICAL_DISKS = 8

// Big ole' array to store the neural values from the whole system
var neuronValuesCurrentStep [TOTAL_NEURONS]float32

// Get blockfiles from the several physical disk directories
// Gets in an order that round-robins their physical disks
func getBlockFiles() []string {
	
	var diskGlobs [][]string
	totalLength := 0

	// Get the globs
	for disk := 0; disk < PHYSICAL_DISKS; disk++ {
		files, _ := filepath.Glob(path.Join(BASE_DIRECTORY,fmt.Sprintf("neuronal%d",disk),"neural_block_*.nb"))
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
func getValueFiles(currentIteration int) []string {

	var valueFilesToProcess []string

	for disk := 0; disk < PHYSICAL_DISKS; disk++ {
		outputFileGlob := path.Join(BASE_DIRECTORY,fmt.Sprintf("neuronal%d",disk),strconv.Itoa(currentIteration),"neural_block_*.nb.output")
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
func writeNeuralBlockWorker(neuralBlockFileChannel <-chan string, wg *sync.WaitGroup) {

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
				connected_neuron := uint64(r.Int63n(int64(TOTAL_NEURONS)))
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
func loadValueFilesToArray(iteration int) {
	//client, _ := hdfs.New("localhost:8020")
	directoryName := fmt.Sprintf("%d/", iteration)
	input := &s3.ListObjectsInput{
		Bucket:  aws.String(NEURAL_DATA_BUCKET),
		Prefix:	 aws.String(directoryName),
		MaxKeys: aws.Int64(1000), // TODO figure out what the limit should be
	}

	svc := s3.New(session.New())

	fmt.Printf("Checking bucket %s/%s\n", NEURAL_DATA_BUCKET, directoryName)
	//fileNames, err := client.ReadDir(directoryName)
	result, _ := svc.ListObjects(input)

	// The session the S3 Downloader will use
	sess := session.Must(session.NewSession())

	// Create a downloader with the session and default options
	downloader := s3manager.NewDownloader(sess)

	//valueBuffer := make([]byte, 4)
	currentNeuron := 0

	fmt.Printf("Processing %d files\n", len(result.Contents))

	// TODO: Randomize?
	for _, objects := range result.Contents {

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

	fmt.Printf("Weights loaded = %d\n", currentNeuron)

}


// keepDoingSomething will keep trying to doSomething() until either
// we get a result from doSomething() or the timeout expires
// Copied from here but there should be a more elegant way
//TODO Connect this to redis
// TODO this seems bad for the call stack, but I don't know any better in go
func pollForDone(redisClient *redis.Client, redisKey string, totalNodes uint64) (bool, error) {

	//timeout := time.After(1000 * time.Second)
	timeout := time.After(20 * time.Second)
	tick := time.Tick(5 * time.Second)


	val, _ := redisClient.Get(redisKey).Result()
	completedNodes,_ := strconv.Atoi(val)
	fmt.Printf("Iteration key = %d\n", completedNodes)

	// Keep trying until we're timed out or got a result or got an error
	for {
		select {
		// Got a timeout! fail with a timeout error
		case <-timeout:
			return false, errors.New("timed out")
		// Got a tick, we should check on doSomething()
		case <-tick:
			if completedNodes >= int(totalNodes) {
				return true, nil
			} else {
				pollForDone(redisClient, redisKey, totalNodes)	
			}
		}
	}
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
// TODO Group the files
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

		outputFileName := fmt.Sprintf("/%d/%s", iteration, path.Base(valueFileGroup[0]))

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

func neuralBlockWorker(iteration int, blockFiles <-chan string, fn activation, wg *sync.WaitGroup) {

	defer wg.Done()

	for fileName := range blockFiles {

		// Some kind of compression cheaper than gzip?
		blockfile, _ := os.OpenFile(fileName, os.O_RDONLY, 0644)
		defer blockfile.Close()

		// Open up file to store neuron output
		//_ = os.Mkdir(fmt.Sprintf("%d", iteration), 0644)
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

	var blocks_per_node uint64 = uint64(math.Ceil((float64(TOTAL_NEURONS)/float64(NEURONS_IN_BLOCK))/float64(TOTAL_NODES)))
	// TODO Account for stragglers in the last node

	fmt.Printf("Running with %d blocks per node\n", blocks_per_node)

	if len(os.Args) < 2 {
		fmt.Println("Missing node number")
		os.Exit(-1)
	}

	node,_ := strconv.Atoi(os.Args[1])
	
	// Worker per CPU
	numberOfWorkers := runtime.NumCPU()

	// The range of neurons this node owns
	var neuronStart uint64 = uint64(node) * blocks_per_node * uint64(NEURONS_IN_BLOCK)
	var neuronEnd uint64 = uint64(node + 1) * blocks_per_node * uint64(NEURONS_IN_BLOCK)

	redisClient := redis.NewClient(&redis.Options{Addr: REDIS_SERVER, Password: "", DB: 0})

	time1 := time.Now()
	// Start by creating random sets of weights and connections, which we will read in and process
	// on subsequent interations
	neuralBlockFileChannel := make(chan string, blocks_per_node)
	for block := 0; block < int(blocks_per_node); block++ {
		neuralBlockFileChannel <- path.Join(BASE_DIRECTORY,blockNumberToDirectoryPrefix(block),fmt.Sprintf("neural_block_%03d_%07d.nb", node, block))
	}
	close(neuralBlockFileChannel)

	// Make workers and write out those files
	var neuralBlockWg sync.WaitGroup
	neuralBlockWg.Add(numberOfWorkers)
	for w := 0; w < numberOfWorkers; w++ {
		go writeNeuralBlockWorker(neuralBlockFileChannel, &neuralBlockWg)
	}
	neuralBlockWg.Wait()
	time2 := time.Now()
	fmt.Printf("Writing out neural block files took %v seconds\n", time2.Sub(time1))

	// Keep track of iterations of the network by number, main execution loop
	for currentIteration := 0; currentIteration <= MAX_ITERATIONS; currentIteration++ {

		// Used to barrier sync the nodes on iteration
		iterationKey := fmt.Sprintf("iteration_%d", currentIteration)

		// Reset if around from previous runs
		if node == 0 {
			redisClient.Set(iterationKey, "0", 0)
		}

		fmt.Printf("Running iteration %d\n\n", currentIteration)
		// TODO how often should we report on block files, there can be hundreds of thousands per node
		//fmt.Printf("Running block file\n\n")

		// Make sure there is a scratch directory for iteration
		for disk := 0; disk < PHYSICAL_DISKS; disk++ {
			os.Mkdir(path.Join(BASE_DIRECTORY,fmt.Sprintf("neuronal%d",disk),strconv.Itoa(currentIteration)), 0777)	
		}

		// On iteration 0, randomly generate neural values, otherwise we need to compute
		if currentIteration == 0 {
			// Initialize the random values
			// This is iteration "0"
			fmt.Printf("Initializing neuron values from %d to %d\n", neuronStart, neuronEnd)
			// Arbitrarily choosing the first physical disk, TODO break this up more later
			writeOutRandomValues(path.Join(BASE_DIRECTORY,"neuronal0","0"), node, neuronStart, neuronEnd)
		} else {
			// Fill channel with block files to process
			// These neural block files contain connections and weights to execute the network
			blockFilesToProcess := getBlockFiles()
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
	        	go neuralBlockWorker(currentIteration, blockFilesChannel, sigmoid, &blockLoadWg)
	    	}
			blockLoadWg.Wait()
			time2 = time.Now()
			fmt.Printf("Processing neural block files took %v seconds\n", time2.Sub(time1))
		}

		// Take the value files to consolidate and upload
		// The value files store the values of each neuron for this iteration
		valueFilesToProcess := getValueFiles(currentIteration)
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
		redisClient.Incr(iterationKey)

		// Poll periodically until everybody's done
		fmt.Printf("Waiting for everybody to be done\n")
		pollForDone(redisClient, iterationKey, TOTAL_NODES)

		// Now iterate through all the files on S3
		fmt.Printf("Loading value files for next iteration\n")
		loadValueFilesToArray(currentIteration)

	}

}