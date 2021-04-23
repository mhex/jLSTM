/**
 * @author Martin Heusel
 *
 * Copyright (C) 2008 Martin Heusel (mhe@bioinf.jku.at),
 *
 * Johannes Kepler University, Linz, Austria
 * Institute of Bioinformatics.
 * The software is maintained by Martin Heusel.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * If you use this library, please cite:
 *
 * @article{SeppHochreiter07152007,
 *   author = {Hochreiter, Sepp and Heusel, Martin and Obermayer, Klaus},
 *   title = {{Fast Model-based Protein Homology Detection without Alignment}},
 *   journal = {Bioinformatics},
 *   volume = {23},
 *   number = {14},
 *   pages = {1728-1736},
 *   doi = {doi:10.1093/bioinformatics/btm247},
 *   year = {2007},
 *   URL = {http://bioinformatics.oxfordjournals.org/cgi/content/abstract/btm247v1},
 *   eprint = {http://bioinformatics.oxfordjournals.org/cgi/reprint/btm247v1}
 * }
 *
 * $Id: JLSTM.java 237 2009-03-03 10:27:03Z mhe $
 *
 */

package at.jku.bioinf.jlstmscopiw;

import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;

import at.jku.bioinf.jlstmscopiw.WeightMatrix;

/**
 * Main class
 *
 * @author mhe
 *
 */
public class JLSTMSCOP {

	// Property file
	static Properties properties;
	// Data containers for the sequences
	static ArrayList<LSTMSequence> seqsTrain;
	static ArrayList<LSTMSequence> seqsTest;
	/* for position coding */
    static int numPositionBins;
    static int numGaussians;
	static int windowSize;
	/* Number of symbols. 20 or 23 for Proteins and 4 for DNA. */
    static int nSymbols;
    /* Number of output units */
    static int numOutputUnits;
    /* Number of LSTM blocks */
    static int numBlocks;
    /* Index of last input unit */
    static int lastInputUnit;
    /* Bias unit */
    static int biasUnit;
    /* Number of inputs for local coding */
    static int numInputsLocalCoding;
    /* Index of last LSTM unit */
    static int lastLSTMUnit;
    /* Number of units */
    static int numUnits;
    /* Blocksize for every block */
    static int[] blockSize;
    /* Range for random init of weights */
    static float initRange;
    /* Biases and initial weights */
    static float[] biasInputGate;
    static float[] biasOutputGate;
    static float[] biasMemInput;
    static float[] weightToOutput;
    static float biasOutput;
    /* GPU device */
    static int gpu;
    /* Batch size */
    static int batchSize;
    /* Learning rate */
    static float alpha;
    /* Upper alpha bound */
    static float upperAlpha;
    /* Error drop p */
    static float p;
    /* Cut dws above/below +/- dwCut */
    static float dwCut;
    /* Dropout rate to the output layer */
    static float dropoutRate;
    /* maximum number of epochs */
    static int maxepochs;
    /* After testnepochs a test is performed */
    static int testnepochs;
    /* After writeweightsafternepochs the weight matrix is written */
    static int writeweightsafternepochs;
    /* Write an additional better readable weight matrix? For debugging
     * and investigation.
     */
    static int humanreadableweightmatrix;
    /* Additional ROCN (e.g. N = 50) to be computed */
    static int rocn;
    /* Weight matrix */
    static WeightMatrix wm;
    static String weightfile;
    static String propertiesfile;
    /* test flag */
    static boolean test;
    static boolean loadweights;
    /* Input matrix */
    static InputMatrix im;
    /* LSTM core */
    static LSTM lstm;
    /* Number of threads */
    static int numThreads;
	/**
	 * Read and set properties from file
	 *
	 */
	static void readProperties() {

	    System.err.println();
        System.err.print("reading properties.. ");

	    // Read properties
	    properties = JLSTMProperties.readProperties(propertiesfile);

        windowSize    = Integer.parseInt(properties.getProperty("windowsize"));
        nSymbols      = Integer.parseInt(properties.getProperty("numberofsymbols"));
        numOutputUnits       = Integer.parseInt(properties.getProperty("numberoutputunits"));
        numBlocks    = Integer.parseInt(properties.getProperty("numbermemorycellblocks"));
        initRange    = Float.parseFloat(properties.getProperty("halfintervallengthforinit"));

        numPositionBins    = Integer.parseInt(properties.getProperty("numberofposbins"));
        numGaussians      = Integer.parseInt(properties.getProperty("numberofgaussians"));

        if (numPositionBins == 0 || numGaussians == 0) {
        	numPositionBins = 0;
        	numGaussians    = 0;
        }

        gpu           = Integer.parseInt(properties.getProperty("gpu"));
        alpha         = Float.parseFloat(properties.getProperty("learningrate"));
        batchSize     = Integer.parseInt(properties.getProperty("batchSize"));
        upperAlpha    = Float.parseFloat(properties.getProperty("upperAlpha"));
        p             = Float.parseFloat(properties.getProperty("p"));
        dwCut         = Float.parseFloat(properties.getProperty("dwCut"));
        dropoutRate   = Float.parseFloat(properties.getProperty("dropoutrate"));
        maxepochs     = Integer.parseInt(properties.getProperty("maxnumberofepochs"));
        testnepochs   = Integer.parseInt(properties.getProperty("performingtestafterNpochs"));
        writeweightsafternepochs  = Integer.parseInt(properties.getProperty("writeweightafterNepochs"));
        humanreadableweightmatrix = Integer.parseInt(properties.getProperty("humanreadableweightmatrix"));
        rocn          = Integer.parseInt(properties.getProperty("rocn"));
        biasOutput = Float.parseFloat(properties.getProperty("outputbias"));
        System.err.println("okay");

	}

	/**
	 * Read data files
	 *
	 */
	static void readData() {

		String fastaFilename  = properties.getProperty("inputdataTraining");
		String labelsFilename = properties.getProperty("labelsTraining");
	    System.err.println();
	    System.err.print("reading data.. ");
	    ReadFasta rf = new ReadFasta(nSymbols, numPositionBins, numGaussians);
	    // Train
	    seqsTrain = rf.read(fastaFilename, labelsFilename, numOutputUnits);
        fastaFilename  = properties.getProperty("inputdataTest");
        labelsFilename = properties.getProperty("labelsTest");
	    // Test
        seqsTest = rf.read(fastaFilename, labelsFilename, numOutputUnits);
        System.err.println("okay");
        System.err.println();
        System.err.println("Train        :" + " " + seqsTrain.size());
        System.err.println("Test         :" + " " + seqsTest.size());

	}

	/**
	 * Read only test data files
	 *
	 */
	static void readDataTest() {

	    System.err.println();
	    System.err.print("reading data for testing.. ");
        String fastaFilename  = properties.getProperty("inputdataTest");
        String labelsFilename = properties.getProperty("labelsTest");
        ReadFasta rf = new ReadFasta(nSymbols, numPositionBins, numGaussians);
	    // Test
        seqsTest = rf.read(fastaFilename, labelsFilename, numOutputUnits);
        System.err.println("okay");
        System.err.println();
        System.err.println("Test             :" + " " + seqsTest.size());

	}

	/**
	 * Calculates automatically biases for all memory cells
	 *
	 */
	static void initNetParams() {

		// Blocksizes
        blockSize = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++) {
            blockSize[i] = 1;
        }
        // Input bias
        biasMemInput = new float[numBlocks];
        Random r = new Random();
        float rangeMin = windowSize / 10.0f;
        float rangeMax = windowSize / 1.0f;
        // Random bias to the memory units
        for (int i = 0; i < numBlocks; i++) {
            biasMemInput[i] =  - (rangeMin + (rangeMax - rangeMin) * r.nextFloat());
        }
        // Inputgate biases
        biasInputGate = new float[numBlocks];
        for (int i = 0; i < numBlocks; i++) {
            biasInputGate[i] = -1;
        }
        // Outputgate biases
        biasOutputGate = new float[numBlocks];
        for (int i = 0; i < numBlocks; i++) {
            biasOutputGate[i] = -1;
        }
        // Output weights alternating
        weightToOutput = new float[numBlocks];
        for (int i = 0; i < numBlocks; i++) {
        	if (i % 2 == 0) {
        		weightToOutput[i] = 0.1f;
        	} else {
        		weightToOutput[i] = 0.1f;
        	}
        }
	}

	/**
	 * Init some stuff
	 *
	 */
	static void init() {

        /* Number of inputs */
        lastInputUnit = nSymbols * windowSize - 1 + numPositionBins;
        /* Bias unit */
        biasUnit = lastInputUnit + 1;
        /* Number of inputs for local coding */
        numInputsLocalCoding = windowSize;
        lastLSTMUnit = biasUnit;
        for (int i = 0; i < numBlocks; i++) {
          lastLSTMUnit += (2 + blockSize[i]);
        }

        numUnits = lastLSTMUnit + numOutputUnits + 1;
        System.err.println();
        int numInputs = lastInputUnit + 1;
        System.err.println("Number of Blocks  : " + numBlocks);
        System.err.println("Window size       : " + windowSize);
        System.err.println("Number of Inputs  : " + numInputs);
        System.err.println("Position bins     : " + numPositionBins);
        System.err.println("Position Gaussians: " + numGaussians);
        System.err.println("Number of Outputs : " + numOutputUnits);
        System.err.println("Number of Units   : " + numUnits);
        System.err.println();
        if (!test) {
            System.err.println("GPU               : " + gpu);
        	System.err.println("C                 : " + upperAlpha);
        	System.err.println("p                 : " + p);
        	System.err.println("Learning rate     : " + alpha);
        	System.err.println("dW cut            : " + dwCut);
        }
        System.err.println("Dropout rate      : " + dropoutRate);
        if (!test) {
        	System.err.println();
        	System.err.println("Stop after        : " + maxepochs + " epochs");
        	System.err.println("Test every        : " + testnepochs + " epochs");
        	System.err.println("Batch size        : " + batchSize);
        }
        System.err.println("ROCn              : " + rocn);
        System.err.println();
        System.err.println("Weight matrix     : " + weightfile);
        System.err.println();

        if (!test) {
        	if (humanreadableweightmatrix != 0) {
        		System.err.println("Additional human readable weight matrix : yes");
        	}
        	System.err.println("Writing weight matrix every              : " + writeweightsafternepochs + " epochs");
        	System.err.println();
        	System.err.println("Number of Threads                        : " + numThreads);
        }

        // Create weightmatrix
        wm = new WeightMatrix(numUnits, lastLSTMUnit, lastInputUnit, biasUnit, numBlocks,
	              biasInputGate, biasOutputGate, biasMemInput,
                weightToOutput, biasOutput,
                blockSize, nSymbols, initRange, alpha);

        wm.setWeightfile(weightfile);

	}

	/**
	 * Parse command line options
	 *
	 * @param args
	 */
	static void parseCommandline(String[] args) {

	    if (args.length < 6 || args.length > 7) {
	        usage();
	    }
	    if (!"-p".equals(args[0])) {
	    	usage();
	    }
	    else {
	    	propertiesfile = args[1];
	    }
	    if (!"-w".equals(args[2])) {
	    	usage();
	    }
	    else {
	    	weightfile = args[3];
	    }

	    if (!"-t".equals(args[4])) {
	    	usage();
	    }
	    else {
	    	numThreads = Integer.parseInt(args[5]);
	    }

	    test        = false;
	    loadweights = false;

	    if (args.length == 7) {
        	if ("-test".equals(args[6])) {
        		test = true;
        	}
        	else if ("-lw".equals(args[6])) {
        		loadweights = true;
        	}
        	else {
        		usage();
        	}
	    }
	}


	static void usage() {

	    System.err.println();
	    System.err.println("Usage: JLSTM -p properties file -w weightfile -t threads [-lw|-test]");
	    System.exit(-1);

	}

	/**
	 * Main
	 *
	 * @param args
	 */
	public static void main(String[] args) {

	    parseCommandline(args);
	    readProperties();
	    initNetParams();
	    if (test) {
	    	readDataTest();
	    }
	    else {
	    	readData();
	    }

        init();

        /* if -lw or -test parameter set then load weights */
        if (test || loadweights) {
        	System.err.println();
        	System.err.print("Reading weight matrix.. ");
        	wm.readWeightMatrix();
        	System.err.println("okay");
        	System.err.println();
        }

        /* test with one thread */

        if (test) {

        	LSTMThread lt = new LSTMThread(numUnits, lastInputUnit, biasUnit, lastLSTMUnit, numInputsLocalCoding,
        			numBlocks, blockSize, nSymbols,
        			gpu,
        			batchSize,
        			alpha, upperAlpha, p, dwCut, dropoutRate,
        			numGaussians, numPositionBins,
        			biasInputGate, biasOutputGate, biasMemInput,
                    weightToOutput, biasOutput,
                    windowSize,
        			rocn,
        			maxepochs,
        			testnepochs,
        			writeweightsafternepochs,
        			null,
        			seqsTest,
        			wm, true,
        			1, 1);

        	/* test and exit */
        	lt.start();
        	try {
        		lt.join();
        	}
        	catch (InterruptedException ie) {
        		System.err.println("Something went wrong joining threads. Should not happen with one test thread.");
        	}
        	System.exit(0);
	    }

        System.err.println();
        System.err.println("Running...");
        System.err.println();

        // Running the threads...

        ArrayList<Thread> threadList = new ArrayList<Thread>();

        for (int thread = 0; thread < numThreads; thread++) {
        	LSTMThread lt = new LSTMThread(numUnits, lastInputUnit, biasUnit, lastLSTMUnit, numInputsLocalCoding,
        			numBlocks, blockSize, nSymbols,
        			gpu,
        			batchSize,
        			alpha, upperAlpha, p, dwCut, dropoutRate,
        			numGaussians, numPositionBins,
        			biasInputGate, biasOutputGate, biasMemInput,
        			weightToOutput, biasOutput,
        			windowSize,
        			rocn,
        			maxepochs,
        			testnepochs,
        			writeweightsafternepochs,
        			seqsTrain,
        			seqsTest,
        			wm, false,
        			numThreads, thread + 1);
        	lt.start();
        	threadList.add(lt);

        	/* Wait a little bit for the next thread */
            /*
        	try {
        		Thread.sleep(1000);
        	}
        	catch (InterruptedException ie){
        		// nothing
        	}
            */
        }

        /* Wait until all threads have finished */
        for (int i = 0; i < threadList.size(); i++) {
        	try {
        		threadList.get(i).join();
        	}
        	catch (InterruptedException ie) {
        		System.err.println("Something went wrong joining threads");
        	}
        }

        /* Start a last test */
        new LSTMThread(numUnits, lastInputUnit, biasUnit, lastLSTMUnit, numInputsLocalCoding,
    			numBlocks, blockSize, nSymbols,
    			gpu,
    			seqsTest.size(),
    			alpha, upperAlpha, p, dwCut, dropoutRate,
    			numGaussians, numPositionBins,
    			biasInputGate, biasOutputGate, biasMemInput,
                weightToOutput, biasOutput,
                windowSize,
    			rocn,
    			maxepochs,
    			testnepochs,
    			writeweightsafternepochs,
    			null,
    			seqsTest,
    			wm, true,
    			numThreads, 1).start();
	}

}
