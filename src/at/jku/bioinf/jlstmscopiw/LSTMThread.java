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
 * $Id: ComputeRoc.java 102 2008-10-08 14:46:53Z mhe $
 *
 */


package at.jku.bioinf.jlstmscopiw;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import Jama.Matrix;


/**
 * LSTM Thread for running a thread
 * Has it's own lstm core and sequence data containers
 * Weight matrix is shared across all threads only the weight update
 * is synchronized.
 *
 * @author mhe
 *
 */
public class LSTMThread extends Thread {

	ArrayList<LSTMSequence> seqsTrain;
	ArrayList<LSTMSequence> seqsTest;

	/* Input matrix */
    private InputMatrix im;

    /* LSTM core */
    private LSTM lstm;

    /* Weight matrix */
    private WeightMatrix wm;

    /* window size */
    private int windowSize;

	/* Number of symbols. 20 or 23 for Proteins and 4 for DNA. */
    private int numSymbols;

    private int numGaussians;

    /* Maximum number of epochs before stop */
    int maxepochs;

    /* After testnepochs a test is performed */
    int testnepochs;

    /* After writeweightsafternepochs the weight matrix is written */
    int writeweightsafternepochs;

    /* Write an additional better readable weight matrix? For debugging
     * and investigation.
     */
    int humanreadableweightmatrix;

    boolean test;

    /* Additional ROCN (e.g. N = 50) to be computed */
    int rocn;

    /* epoch is shared across all threads */
    static int epoch;

    /* Internal epoch used for flagging trained sequences */
    private int internalEpoch;

    int batchSize;

    ReentrantReadWriteLock rwl = new ReentrantReadWriteLock();
    private final Lock wlock = rwl.writeLock();

    QPSolverBoxConstraints qps;

    SMO smo;

    float[] deltaLoss;
    float[] p;

    float upperAlpha;

    float dwCut;

    int threadNr;

    int numThreads;

    public LSTMThread(int numUnits, int lastInputUnit, int biasUnit, int lastLSTMUnit, int numInputsLocalCoding,
            int numBlocks, int[] blockSize, int numSymbols,
            int gpu,
            int batchSize,
            float alpha, float upperAlpha, float pVal, float dwCut,
            float dropoutRate,
            int numGaussians, int numPositionBins,
            float[] biasInputGate, float[] biasOutputGate, float[] biasMemUnit,
            float[] weightToOutput, float outputBias,
            int windowSize,
            int rocn,
            int maxepochs,
            int testnepochs,
            int writeweightsafternepochs,
            ArrayList<LSTMSequence> seqsTrain,
            ArrayList<LSTMSequence> seqsTest,
            WeightMatrix wm, boolean test,
            int numThreads, int threadNr) {

    	this.windowSize = windowSize;
    	this.numGaussians = numGaussians;
    	this.writeweightsafternepochs = writeweightsafternepochs;
    	this.testnepochs = testnepochs;
    	this.test = test;

        //int numSeqsTrain   = (seqsTrain != null) ? seqsTrain.size() : 0;
        int numSeqsTest    = seqsTest.size();
    	int numSeqsTrain = batchSize;

        if (!test) {
        	this.seqsTrain  = (ArrayList<LSTMSequence>)seqsTrain.clone();
        }

        this.seqsTest       = (ArrayList<LSTMSequence>)seqsTest.clone();
        this.wm             = wm;
        this.batchSize      = batchSize;
        this.upperAlpha     = upperAlpha;
        this.dwCut          = dwCut;
        this.internalEpoch  = 1;
        this.maxepochs      = maxepochs;
        this.numThreads     = numThreads;
        this.threadNr       = threadNr;

        // Create QP Solver
        qps = new QPSolverBoxConstraints(batchSize, numUnits, numUnits, 0,  upperAlpha, gpu);

        // Create SMO Solver
        //smo = new SMO(seqsTrain.size(), numUnits, numUnits);

        deltaLoss = new float[batchSize];
        p         = new float[batchSize];

        Arrays.fill(deltaLoss, pVal);
        Arrays.fill(p, pVal);

    	// Create inputmatrix
        this.im = new InputMatrix();

        // Make LSTM network
        if (test) {
        	lstm = new LSTM(numUnits, lastInputUnit, biasUnit, lastLSTMUnit, numInputsLocalCoding,
                     numBlocks, blockSize, numSymbols, alpha, dropoutRate,
                     numGaussians, numPositionBins,
                     0, numSeqsTest,
                     rocn,
                     wm, threadNr);
        }
        else {
        	lstm = new LSTM(numUnits, lastInputUnit, biasUnit, lastLSTMUnit, numInputsLocalCoding,
        			numBlocks, blockSize, numSymbols, alpha, dropoutRate,
        			numGaussians, numPositionBins,
        			batchSize, numSeqsTest,
        			rocn,
        			wm, threadNr);
        	lstm.init(biasInputGate, biasOutputGate,biasMemUnit, weightToOutput, outputBias);
        }
    }

	/**
	 * Training
	 *
	 * @param epoch
	 */
	void training(int epoch, boolean trainAll) {

		// For all training sequences
	    //int numSeqsTrain = seqsTrain.size();
        for (int seqNr = 0; seqNr < batchSize; seqNr++) {
        	LSTMSequence seq = seqsTrain.get(seqNr);
        	// Skip this sequence flagged without error
        	if (!trainAll) {
        		if (!seq.getToTrain()) {
        		    continue;
        		}
        	}
        	// Create input matrix
            im.computeInputs(seq, windowSize, numSymbols, numGaussians);
            // Set  input index and input vector in the lstm core
            lstm.setInputIndex(im.getInputIndex());
            lstm.setInputVector(im.getInputVector());
            // Set label
            lstm.setTargets(seq.getTargets());
            // For all elements in the current sequence i except the last one
            int j;
            for (j = 0; j < seq.getLocalCodings().length - 1; j++) {
            	lstm.cloneWeightMatrix();
                lstm.forwardPass(j, false, false);
                lstm.derivatives(j);
                lstm.timeforward();
            }
            // The last element has a target
            lstm.cloneWeightMatrix();
            lstm.shuffleDropoutMask();
            lstm.forwardPass(j, true, true);
            boolean isError = lstm.compError();
            //System.out.println("Thread " + threadNr + " " + isError + " " + i + " " + internalEpoch);
            //seq.setToTrain(isError);
            lstm.backwardPass(j);
            /* For QP solver */
            lstm.fillDWMatrix(seqNr);
            lstm.compCrossEntropyError(seqNr, true);
            lstm.compRanks(seqNr, true);
            lstm.reset();
        }

        lstm.printError(epoch, seqsTrain, true);

        /* Calculate dw kernel */
        Matrix dwK = qps.calcDWKernelGPU(lstm.getDwMatrix(), threadNr);
        //Matrix dwK = qps.calcDWKernelCPU(lstm.getDwMatrix(), threadNr);

        float min = Integer.MAX_VALUE;
        float max = Integer.MIN_VALUE;

        float sumCEE = 0;

        for (int i = 0; i < deltaLoss.length; i++) {
            float crossEntropyError = lstm.getCrossEntropyErrorsTrain()[i];
            min = (crossEntropyError < min) ? crossEntropyError : min;
            max = (crossEntropyError > max) ? crossEntropyError : max;
            sumCEE += crossEntropyError;
            float crossEntropyErrorLim = (crossEntropyError > 4.0) ? 4.0f : crossEntropyError;
            deltaLoss[i] = p[i] * crossEntropyErrorLim;
        }

        System.out.format("Thread %d Crossentropy Error: min: %.6f max: %.6f mean %.6f", threadNr, min, max, sumCEE/deltaLoss.length);
        System.out.println();

        /* Solve qp */
        float[] alphas = qps.solveJO(dwK, deltaLoss, 0, upperAlpha, threadNr);
        //float[] alphas = smo.solve(lstm.getDwMatrix(), deltaLoss, 0, upperAlpha, threadNr);
        //float[] alphas = new float[batchSize];

        min = Integer.MAX_VALUE;
        max = Integer.MIN_VALUE;

        float sumAlphas = 0;

        for (float alpha : alphas) {
            min = Math.min(alpha, min);
            max = Math.max(alpha, max);
            sumAlphas += alpha;
        }

        System.out.format("Thread %d alphas: min: %.6f max: %.6f mean %.6f", threadNr, min, max, sumAlphas/alphas.length);
        System.out.println();

        float[][] dW = wm.calcBatchdW(lstm.getDwMatrix(), alphas, dwCut, threadNr);

        /* write lock the weight update */
        wlock.lock();
        //wm.weightUpdateIW(dW, threadNr);
        wm.weightUpdate(dW);
        wlock.unlock();

        lstm.setEpoch_err(0);
        lstm.setFalsepos(0);
        lstm.setFalseneg(0);
        lstm.resetCrossEntropyErrors();

        System.out.println();

	}

	/**
	 * Test
	 *
	 * @param epoch
	 */
	void test(int epoch) {

		System.out.println("Thread: " + threadNr);
		System.out.println("epoch:  " + epoch + "  Start test..");
		System.out.println();

    	// For all test sequences
    	for (int i = 0; i < seqsTest.size(); i++) {

    		LSTMSequence seq = seqsTest.get(i);
    		im.computeInputs(seq, windowSize, numSymbols, numGaussians);
    		lstm.setInputIndex(im.getInputIndex());
            lstm.setInputVector(im.getInputVector());
    		lstm.setTargets(seqsTest.get(i).getTargets());
    		lstm.cloneWeightMatrix();
    		lstm.reset();
    		int j;
    		for (j = 0; j < seqsTest.get(i).getLocalCodings().length - 1; j++) {
    			lstm.forwardPass(j, false, false);
    			lstm.timeforward();
    		}
    		lstm.forwardPass(j, true, false);
    		lstm.compError();
    		lstm.compCrossEntropyError(i, false);
    		lstm.compRanks(i, false);
    		lstm.reset();
    	}

    	lstm.printError(epoch, seqsTest, false);

    	float min = Integer.MAX_VALUE;
        float max = Integer.MIN_VALUE;
        float sumCEE = 0;

        for (int i = 0; i < seqsTest.size(); i++) {
            float crossEntropyError = lstm.getCrossEntropyErrorsTest()[i];
            min = Math.min(crossEntropyError, min);
            max = Math.max(crossEntropyError, max);
            sumCEE += crossEntropyError;
        }

        System.out.format("TEST: Thread %d Crossentropy Error: min: %.6f max: %.6f mean %.6f", threadNr, min, max, sumCEE/seqsTest.size());
        System.out.println();
        System.out.println();

    	lstm.setEpoch_err(0);
    	lstm.setFalsepos(0);
    	lstm.setFalseneg(0);

	}


	/* (non-Javadoc)
	 * @see java.lang.Thread#run()
	 */
	@Override public void run() {

		int myEpoch;

		if (test) {
			test(epoch);
			return;
		}

		Random r = new Random();
		Lock lock = new ReentrantLock();
		while (epoch <= maxepochs - 1) {
			lock.lock();
            epoch++;
            myEpoch = epoch;
            lock.unlock();
            // Shuffle data set
            Collections.shuffle(seqsTrain, r);
            // Train all flag
            boolean trainAll = true;
            // Training
            training(myEpoch, trainAll);
            // Write weight matrix
            if ((myEpoch % writeweightsafternepochs == 0 && myEpoch > 0) || myEpoch == maxepochs) {
            	System.out.println("Thread: " + threadNr);
            	System.out.println("write weightmatrix");
            	System.out.println();
                wm.writeWeightMatrix();
                if (humanreadableweightmatrix != 0) {
                    wm.writeReadableWeightMatrix();
                }
            }
            // Test after n epochs
            if ((myEpoch % testnepochs == 0 && myEpoch > 0) || myEpoch == maxepochs) {
            	test(myEpoch);
            }
            internalEpoch++;
        }
	}
}
