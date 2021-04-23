/**
 * @author Martin Heusel
 * <p>
 * Copyright (C) 2008 Martin Heusel (mhe@bioinf.jku.at),
 * <p>
 * Johannes Kepler University, Linz, Austria
 * Institute of Bioinformatics.
 * The software is maintained by Martin Heusel.
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * If you use this library, please cite:
 * @article{SeppHochreiter07152007, author = {Hochreiter, Sepp and Heusel, Martin and Obermayer, Klaus},
 * title = {{Fast Model-based Protein Homology Detection without Alignment}},
 * journal = {Bioinformatics},
 * volume = {23},
 * number = {14},
 * pages = {1728-1736},
 * doi = {doi:10.1093/bioinformatics/btm247},
 * year = {2007},
 * URL = {http://bioinformatics.oxfordjournals.org/cgi/content/abstract/btm247v1},
 * eprint = {http://bioinformatics.oxfordjournals.org/cgi/reprint/btm247v1}
 * }
 * <p>
 * $Id: LSTM.java 227 2009-02-13 11:00:21Z mhe $
 */


package at.jku.bioinf.jlstmscopiw;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.ArrayList;

/**
 * This is the core of the LSTM. Computes forward, backward pass and errors.
 *
 * @author mhe
 *
 */
public class LSTM {

    int numUnits;
    int numOutputUnits;
    int lastInputUnit;
    int biasUnit;
    int lastMemoryUnit;
    int firstOutputUnit;
    int numInputsLocalCoding;
    int numBlocks;
    int[] blockSize;

    int nSymbols;

    int numGaussians;
    int numPositionBins;

    float[] activationOld;
    float[] activationNew;
    float[] activationOutputGate;
    float[] activationInputGate;
    float[][] activationCell;
    float[][] g;
    float[][] h;
    float[][] memoryState;
    float[][][] sc;
    float[][][] si;

    float[] errorSignalOutput;
    float[] errorSignalOutputGate;
    float[][] errorSignalMemoryOut;
    float[][] errorSignalMemoryState;


    float[] error;
    float[] targets;
    float[] crossEntropyErrorsTrain;
    float[] crossEntropyErrorsTest;

    float sequenceError;
    float sequenceCorrect;
    float epochError;

    int falsepos;
    int falseneg;

    int[] rankTest;
    int[] rankTrain;
    float rankMean;

    Integer[] rankTestSorted;
    Integer[] rankTrainSorted;

    float aucVal;
    float aucNVal;

    int numSeqsTrain;
    int numSeqsTest;

    float alpha;

    float dropoutRate;

    Random rand;

    boolean[] dropoutMask;

    int rocn;

    float[][] inputVector;
    short[][] inputIndex;

    WeightMatrix wm;

    float[][] weightMatrixLocal;
    float[][] weightMatrixGlobal;

    float[][] dW;

    // dWs for all samples in an epoch
    float[][][] dwMatrix;

    int threadNr;

    public LSTM(int numUnits, int lastInputUnit, int biasUnit, int lastMemoryUnit, int numInputsLocalCoding,
                int numBlocks, int[] blockSize, int nSymbols, float alpha, float dropoutRate,
                int numGaussians, int numPosBins,
                int numSeqsTrain, int numSeqsTest,
                int rocn,
                WeightMatrix wm,
                int threadNr) {

        this.numUnits = numUnits;
        this.lastInputUnit = lastInputUnit;
        this.biasUnit = biasUnit;
        this.lastMemoryUnit = lastMemoryUnit;
        this.firstOutputUnit = lastMemoryUnit + 1;
        this.numOutputUnits = numUnits - firstOutputUnit;
        this.numInputsLocalCoding = numInputsLocalCoding;
        this.numBlocks = numBlocks;
        this.blockSize = blockSize;

        this.nSymbols = nSymbols;

        this.numGaussians = numGaussians;
        this.numPositionBins = numPosBins;

        this.numSeqsTrain = numSeqsTrain;
        this.numSeqsTest = numSeqsTest;

        this.rankTest = new int[numSeqsTest];
        this.rankTrain = new int[numSeqsTrain];

        this.rankTestSorted = new Integer[numSeqsTest];
        this.rankTrainSorted = new Integer[numSeqsTrain];

        this.alpha = alpha;

        this.dropoutRate = dropoutRate;

        this.dropoutMask = initDropoutMask();

        this.rocn = rocn;

        this.wm = wm;

        this.weightMatrixGlobal = wm.getWeightMatrix();

        this.threadNr = threadNr;

        // Init matrices

        dW = new float[numUnits][numUnits];

        dwMatrix = new float[numSeqsTrain][numUnits][numUnits];

        activationOld = new float[numUnits + 1];
        activationNew = new float[numUnits + 1];

        activationOutputGate = new float[numBlocks];
        activationInputGate = new float[numBlocks];
        activationCell = new float[numBlocks][1]; // TODO: change 5 to maximum blocksize
        g = new float[numBlocks][1];
        h = new float[numBlocks][1];
        memoryState = new float[numBlocks][1];

        si = new float[numBlocks][5][numUnits + 1];
        sc = new float[numBlocks][5][numUnits + 1];

        errorSignalOutput = new float[numUnits + 1];
        errorSignalOutputGate = new float[numBlocks];
        errorSignalMemoryOut = new float[numBlocks][1];
        errorSignalMemoryState = new float[numBlocks][1];

        error = new float[numOutputUnits];

        crossEntropyErrorsTrain = new float[numSeqsTrain];
        crossEntropyErrorsTest = new float[numSeqsTest];

        //this.roclist = new ArrayList<RocData>(numseqstrain);

        this.sequenceError = 0;
        this.sequenceCorrect = 1;
        this.epochError = 0;

        this.falsepos = 0;
        this.falseneg = 0;

        reset();

    }

    boolean[] initDropoutMask() {
        dropoutMask = new boolean[numBlocks];
        int dropout = Math.round(numBlocks * this.dropoutRate);
        for (int i = 0; i < dropout; i++) {
            this.dropoutMask[i] = true;
        }
        return dropoutMask;
    }

    /**
     * Fast exponential approximation
     *
     * {@link http://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/}
     *
     * @param val
     * @return
     */
    public float fastExp(float val) {
        final long tmp = (long) (1512775 * val + (1072693248 - 60801));
        return (float) Double.longBitsToDouble(tmp << 32);
    }

    void compRanks(int seqNr, boolean isTrain) {
        int rank = this.numOutputUnits;
        int posIndex = 0;

        for (int i = firstOutputUnit, j = 0; i < numUnits; i++, j++) {
            if (targets[j] == 1.0) {
                posIndex = i;
                break;
            }
        }

        float actPos = activationNew[posIndex];

        /* compute error */
        for (int i = firstOutputUnit; i < numUnits; i++) {
            if (actPos > activationNew[i]) rank--;
            //System.out.println(j + " act: " + activationNew[i] + " targ: " + targets[j]);
        }

        if (isTrain) {
            this.rankTrain[seqNr] = rank;
        } else {
            this.rankTest[seqNr] = rank;
        }
    }


    /**
     * Compute the error
     */
    boolean compError() {

        boolean seqErr = false;
        float err = 0;
        int posIdx = -1;

        /* compute error */
        for (int i = firstOutputUnit, j = 0; i < numUnits; i++, j++) {
            //error[j] = activationNew[i] - targets[j];
            error[j] = targets[j] - activationNew[i];
            if (targets[j] == 1.0) {
                posIdx = i;
            }
            err += error[j] * error[j];
        }

        /* Calculate top-1 error */
        for (int i = firstOutputUnit; i < numUnits; i++) {
            if (i != posIdx) {
                if (activationNew[i] >= activationNew[posIdx]) {
                    seqErr = true;
                    break;
                }
            }
        }


        err = err / numOutputUnits;
        sequenceError += err;
        epochError += err;

        if (seqErr) {
            falsepos++;
            falseneg++;
        }

		/*
        if ((sequenceCorrect == 1) && (Math.abs(error[0]) > Math.abs(targetpositive - targetnegative)/2)) {

          sequenceCorrect = 0;

          if (this.ispositive) {
              falseneg++;
          }
          else {
              falsepos++;
          }

        }
		 */

        return (seqErr);

    }

    public void compCrossEntropyError(int seqNr, boolean isTrain) {

        float crossEntropyError = 0;

        for (int i = firstOutputUnit, j = 0; i < numUnits; i++, j++) {
            crossEntropyError -= Math.log(activationNew[i]) * targets[j];
        }

        if (isTrain) {
            crossEntropyErrorsTrain[seqNr] = crossEntropyError;
        } else {
            crossEntropyErrorsTest[seqNr] = crossEntropyError;
        }

    }

    private float compRankMean(int[] rank, int numSeqs) {

        float sum = 0;

        for (int i = 0; i < numSeqs; i++) {
            sum += rank[i];
        }

        return (sum / numSeqs);

    }

    private void compAUC(int[] rank, Integer[] rankSorted, int numSeqs) {

        aucVal = 0;
        aucNVal = 0;

        for (int i = 0; i < numSeqs; i++) {
            rankSorted[i] = rank[i];
        }

        Arrays.sort(rankSorted, Collections.reverseOrder());

        for (int i = 0; i < numSeqs; i++) {

            float sense;

            if (i < rocn) {
                sense = (rocn - rankSorted[i] + 1) / (float) rocn;
                //fpr   = ((int)rankTestSorted[i] - 1) / (float) rocn;
                aucNVal += sense;
            }

            sense = (numOutputUnits - rankSorted[i] + 1) / (float) this.numOutputUnits;
            //fpr   = (rankTestSorted[i] - 1) / (float)numOutputUnits;
            aucVal += sense;

        }

        aucVal /= numSeqs;
        aucNVal /= numSeqs;

    }

    /**
     * Output of net performance (MSE, false-positives and false-negatives)
     *
     * @param isTrain
     */
    public void printError(int epoch, ArrayList<LSTMSequence> seqsArray, boolean isTrain) {

        int numSeqs;
        String what = "";
        String pre = "";
        float fnp = 1.0f;

        if (isTrain) {
            numSeqs = numSeqsTrain;
            what = "training";
            pre = "";
            rankMean = compRankMean(rankTrain, numSeqs);
            compAUC(rankTrain, rankTrainSorted, numSeqs);
        } else {
            numSeqs = numSeqsTest;
            what = "test";
            pre = "TEST: ";
            rankMean = compRankMean(rankTest, numSeqs);
            compAUC(rankTest, rankTestSorted, numSeqs);
            //cr.computeAUC(roclist, numpostest, numnegtest, rocn);
        }

        int countTop1 = 0;
        int countTop5 = 0;
        int[] ranks;

        if (isTrain) {
            ranks = rankTrain;
        } else {
            ranks = rankTest;
        }

        for (int i = 0; i < numSeqs; i++) {
            if (ranks[i] > 1) {
                countTop1++;
            }
            if (ranks[i] > 5) {
                countTop5++;
            }
        }

        float top1Error = countTop1 / (float) numSeqs * 100.0f;
        float top5Error = countTop5 / (float) numSeqs * 100.0f;
        float[] crossEntropyErrors = (isTrain) ? crossEntropyErrorsTrain : crossEntropyErrorsTest;
        float min = Integer.MAX_VALUE;
        float max = Integer.MIN_VALUE;
        float sumCEE = 0;

        for (int i = 0; i < numSeqs; i++) {
            float crossEntropyError = crossEntropyErrors[i];
            min = Math.min(crossEntropyError, min);
            max = Math.max(crossEntropyError, max);
            sumCEE += crossEntropyError;
        }

        float crossEntropyErrorMean = sumCEE / numSeqs;
        int[] labels = new int[numSeqs];

        for (int seqNr = 0; seqNr < numSeqs; seqNr++) {
            LSTMSequence seq = seqsArray.get(seqNr);
            float[] seqTargets = seq.getTargets();
            for (int j = 0; j < seqTargets.length; j++) {
                if (seqTargets[j] == 1.0) labels[seqNr] = j + 1;
            }
        }

		/*
        float auc = cr.getAuc();
        float aucn = cr.getAucn();
		 */

        fnp = (float) falseneg / (float) numSeqs;
        //float fpp = (numneg != 0) ? (float) falsepos / (float) numneg : 0;
        float mse = epochError / (1.0f * numSeqs);

        BufferedWriter out = null;

        try {

            out = new BufferedWriter(new FileWriter("out.txt", true));

            out.write("epoch: " + epoch);
            out.newLine();
            out.write(pre + "MSE ");
            out.write(new DecimalFormat("#.######").format(mse));
            out.newLine();

            out.write(pre + "errors:");
            out.write(falseneg + " (out of " + numSeqs + " " + what + " examples) ");
            out.write(new DecimalFormat("#0.0").format(fnp * 100) + "%");
            out.newLine();

            out.write(pre + "Top 1 Error: ");
            out.write(new DecimalFormat("#0.00").format(top1Error));
            out.newLine();
            out.write(pre + "Top 5 Error: ");
            out.write(new DecimalFormat("#0.00").format(top5Error));
            out.newLine();

            out.write(pre + "Crossentropy Loss: ");
            out.write(new DecimalFormat("#0.00").format(crossEntropyErrorMean));
            out.newLine();

			/*
          out.write(pre + "false-positive:");
          out.write(falsepos + " (out of " + numneg + " negative " + what + " examples) ");
          out.write(new DecimalFormat("#0.0").format(fpp * 100) + "%");
          out.newLine();
			 */
            out.write(pre + "ROC ");
            out.write(new DecimalFormat("#0.000").format(aucVal));
            out.newLine();
            out.write(pre + "ROC" + rocn + " ");
            out.write(new DecimalFormat("#0.000").format(aucNVal));
            out.newLine();

            if (!isTrain) {
                out.write(pre + "Ranks: ");
                for (int i = 0; i < this.numSeqsTest; i++) {
                    out.write(this.rankTest[i] + " ");
                }
                out.newLine();
                out.write(pre + "CEL: ");
                for (int i = 0; i < this.numSeqsTest; i++) {
                    out.write(this.crossEntropyErrorsTest[i] + " ");
                }
                out.newLine();
                out.write(pre + "Label: ");
                for (int i = 0; i < this.numSeqsTest; i++) {
                    out.write(labels[i] + " ");
                }
                out.newLine();
            } else {
                out.write("Ranks: ");
                for (int i = 0; i < this.numSeqsTrain; i++) {
                    out.write(this.rankTrain[i] + " ");
                }
                out.newLine();
                out.write("CEL: ");
                for (int i = 0; i < this.numSeqsTrain; i++) {
                    out.write(this.crossEntropyErrorsTrain[i] + " ");
                }
                out.newLine();
                out.write("Label: ");
                for (int i = 0; i < this.numSeqsTrain; i++) {
                    out.write(labels[i] + " ");
                }
                out.newLine();
            }
            out.write(pre + "Rank mean: " + new DecimalFormat("#0.00").format(rankMean));
            out.newLine();
            out.newLine();
            out.close();

        } catch (IOException e) {
            System.err.println(e);
        } finally {
            try {
                if (out != null) out.close();
            } catch (IOException e) {
            }
        }

        System.out.println("Thread: " + threadNr);
        System.out.println("epoch:  " + epoch);
        System.out.printf("%sMSE:%f\n", pre, mse);
        System.out.printf("%serrors:%d (out of %d %s examples) %.1f%%\n", pre, falseneg, numSeqs, what, fnp * 100);

        System.out.print(pre + "Top 1 Error: ");
        System.out.print(new DecimalFormat("#0.00").format(top1Error));
        System.out.println();
        System.out.print(pre + "Top 5 Error: ");
        System.out.print(new DecimalFormat("#0.00").format(top5Error));
        System.out.println();

		/*
        System.out.printf("%sfalse-positive:%d (out of %d negative %s examples) %.1f%%\n", pre, falsepos, numneg, what, fpp * 100);
		 */
        System.out.printf("%sROC %.3f\n", pre, aucVal);
        System.out.printf("%sROC" + rocn + " %.3f\n", pre, aucNVal);


        if (!isTrain) {
            System.out.print("Ranks: ");
            for (int i = 0; i < this.numSeqsTest; i++) {
                System.out.print(this.rankTest[i] + " ");
            }
            System.out.println();
        }
        System.out.printf(pre + "Rank mean: %.2f", rankMean);
        System.out.println();

    }

    /**
     * Timestep
     *
     */
    public void timeforward() {
        if (numUnits >= 0) {
            System.arraycopy(activationNew, 0, activationOld, 0, numUnits);
        }
        sequenceError = 0;
        sequenceCorrect = 1;
    }

    public void cloneWeightMatrix() {
        this.weightMatrixLocal = this.weightMatrixGlobal.clone();
    }

    public void shuffleDropoutMask() {
        Collections.shuffle(Arrays.asList(this.dropoutMask));
    }

    /**
     * Forward pass
     *
     * @param element
     * @param targ
     * @param train
     */
    public void forwardPass(int element, boolean targ, boolean train) {

        /* ### memory cells ### */
        int i = biasUnit + 1;

        for (int u = 0; u < numBlocks; u++) {

            float sumInputGate;
            float sumOutputGate;
            float sum;

            sumInputGate = 0;
            sumOutputGate = 0;
            /* input */

			/*
            for (int j=0;j<in_nn_mod;j++) {
                sumInputGate  += weightMatrixLocal[i][inputIndex[element][j]] * inputVector[element][j];
            }

            for (int j=0;j<in_nn_mod;j++) {
                sumOutputGate += weightMatrixLocal[i + 1][inputIndex[element][j]] * inputVector[element][j];
            }
			 */

            /* lstm */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {
                sumInputGate += weightMatrixLocal[i][j] * activationOld[j];
                sumOutputGate += weightMatrixLocal[i + 1][j] * activationOld[j];

            }

            /* input gates */
            activationInputGate[u] = 1 / (1 + fastExp(-sumInputGate));
            activationNew[i] = activationInputGate[u];

            /* output gates */
            i++;
            activationOutputGate[u] = 1 / (1 + fastExp(-sumOutputGate));
            activationNew[i] = activationOutputGate[u];

            /* peep hole */
            //sumOutputGate -= S[u][0];

            /* uth memory cell block */
            for (int v = 0; v < blockSize[u]; v++) {
                /* activation of function g of vth memory cell of block u  */
                i++;
                if (train) {
                    if (!dropoutMask[u]) {
                        sum = weightMatrixLocal[i][biasUnit]; // bias
                        for (int j = 0; j < numInputsLocalCoding; j++) {
                            sum += weightMatrixLocal[i][inputIndex[element][j]]; // * inputVector[element][j];
                        }
                        /* position infos (if any) */
                        for (int j = 0; j < numGaussians; j++) {
                            //if (targ) {
                            //System.out.print(element + " " + i + " " + j + " ");
                            //System.out.println(" " + j + inputIndex[element][numInputsLocalCoding + j] + " " + inputVector[element][numInputsLocalCoding + j]);
                            //}
                            sum += weightMatrixLocal[i][inputIndex[element][numInputsLocalCoding + j]] * inputVector[element][numInputsLocalCoding + j];
                        }

                        /* peep hole */
                        //sum -= S[u][v];
                        //sumblock[u] = sum;

                        if (sum > 20) {
                            g[u][v] = 2.0f;
                        } else {
                            if (sum < -20) {
                                g[u][v] = 1e-9f;
                            } else {
                                g[u][v] = 2.0f / (1 + fastExp(-sum));
                            }
                        }
                    } else {
                        g[u][v] = 0;
                    }

                } else {  /* test */
                    sum = weightMatrixLocal[i][biasUnit]; // bias
                    for (int j = 0; j < numInputsLocalCoding; j++) {
                        sum += weightMatrixLocal[i][inputIndex[element][j]]; // * inputVector[element][j];
                    }
                    /* position infos (if any) */
                    for (int j = 0; j < numGaussians; j++) {
                        //if (targ) {
                        //System.out.print(element + " " + i + " " + j + " ");
                        //System.out.println(" " + j + inputIndex[element][numInputsLocalCoding + j] + " " + inputVector[element][numInputsLocalCoding + j]);
                        //}
                        sum += weightMatrixLocal[i][inputIndex[element][numInputsLocalCoding + j]] * inputVector[element][numInputsLocalCoding + j];
                    }
                    sum *= (1 - dropoutRate);
                    /* peep hole */
                    //sum -= S[u][v];
                    //sumblock[u] = sum;
                    if (sum > 20) {
                        g[u][v] = 2.0f;
                    } else {
                        if (sum < -20) {
                            g[u][v] = 1e-9f;
                        } else {
                            g[u][v] = 2.0f / (1 + fastExp(-sum));
                        }
                    }
                }

                /* update internal state  */
                memoryState[u][v] += activationInputGate[u] * g[u][v];

                /* activation function h */
                if (memoryState[u][v] > 20.0) {
                    h[u][v] = 1.0f;
                } else {
                    h[u][v] = 2.0f / (1 + fastExp(-memoryState[u][v])) - 1.0f;
                }

                /* activation of vth memory cell of block u */
                activationCell[u][v] = h[u][v] * activationOutputGate[u];
                activationNew[i] = activationCell[u][v];
            }
            i++;
        }

        /* ### output units activation ### */
        if (targ) /* only if target for this input */ {
            for (int k = firstOutputUnit; k < numUnits; k++) {
                /* bias */
                float sum = weightMatrixLocal[k][biasUnit];
                /* memory cells input */
                i = biasUnit + 1;
                for (int u = 0; u < numBlocks; u++) {
                    i++;
                    for (int v = 0; v < blockSize[u]; v++) {
                        i++;
                        // dropout
                        sum += weightMatrixLocal[k][i] * activationNew[i];
                    }
                    i++;
                }
                /* activation */
                activationNew[k] = sum;
            }

            float maxSum = 0;
            for (int k = firstOutputUnit; k < numUnits; k++) {
                maxSum = Math.max(maxSum, activationNew[k]);
            }

            float tmp;
            float sumExp = 0;

            for (int k = firstOutputUnit; k < numUnits; k++) {
                tmp = activationNew[k];
                tmp -= maxSum;
                tmp = (tmp < -20) ? 1e-9f : fastExp(tmp);
                sumExp += tmp;
            }

            sumExp = (float) Math.log(sumExp);

            for (int k = firstOutputUnit; k < numUnits; k++) {
                tmp = activationNew[k] - maxSum - sumExp;
                activationNew[k] = (tmp < -20) ? 1e-9f : fastExp(tmp);
            }
        }
    }


    /**
     * Backward pass
     *
     * @param element
     */
    public void backwardPass(int element) {

        /* output units */
        for (int k = firstOutputUnit, j = 0; k < numUnits; k++, j++) {

            // logistic activation
            //errorSignalOutput[k] = error[j] * (1.0 - activationNew[k]) * activationNew[k]; // log
            // Softmax activation
            errorSignalOutput[k] = error[j];
            /* weight update contribution */
            //float tmp = errorSignalOutput[k];

            int i = biasUnit + 1;

            for (int u = 0; u < numBlocks; u++) {
                i++;
                for (int v = 0; v < blockSize[u]; v++) {
                    i++;
                    dW[k][i] += errorSignalOutput[k] * activationNew[i];
                }
                i++;
            }
            /* bias to output unit */
            dW[k][biasUnit] += errorSignalOutput[k];
        }

        /* error to memory cells ec[][] and internal states es[][] */

        int i = biasUnit + 1;

        /* memory state */
        for (int u = 0; u < numBlocks; u++) {
            i++;
            for (int v = 0; v < blockSize[u]; v++) {
                i++;
                float sum = 0;
                for (int k = firstOutputUnit; k < numUnits; k++) {
                    sum += weightMatrixLocal[k][i] * errorSignalOutput[k];
                }
                errorSignalMemoryState[u][v] = 0.5f * activationOutputGate[u] * (1.0f + h[u][v]) * (1.0f - h[u][v]) * sum;
                errorSignalMemoryOut[u][v] = sum;
            }
            i++;
        }

        /* output gates */
        for (int u = 0; u < numBlocks; u++) {
            float sum = 0;
            for (int v = 0; v < blockSize[u]; v++) {
                sum += h[u][v] * errorSignalMemoryOut[u][v];
            }
            errorSignalOutputGate[u] = sum * (1.0f - activationOutputGate[u]) * activationOutputGate[u];
        }

        /* Derivatives of the internal state */
        derivatives(element);

        /* updates for weights to input and output gates and memory cells */
        i = biasUnit + 1;
        for (int u = 0; u < numBlocks; u++) {
            /* input gates */
            /* input */
			/*
            for (int j=0;j<in_mod_b;j++) {
                float sum = 0;
                for (int v=0;v<blockSize[u];v++) {
                    sum += es[u][v]*si[u][v][j];
                }
                DW[i][j] += sum;
            }
			*/
            /* lstm */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {
                float sum = 0;
                for (int v = 0; v < blockSize[u]; v++) {
                    sum += errorSignalMemoryState[u][v] * si[u][v][j];
                }
                //dW[i][j] += alpha * sum;
                dW[i][j] += sum;
            }

            /*output gates */
            i++;
            //float tmp = alpha * errorSignalOutputGate[u];
            float tmp = errorSignalOutputGate[u];
            /* input */
			/*
            for (int j=0;j<in_mod;j++) {
                DW[i][j] += tmp * yk_old[j];
            }
			*/
            /* lstm */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {
                dW[i][j] += tmp * activationOld[j];
            }

            /* memory states */
            for (int v = 0; v < blockSize[u]; v++) {
                i++;
                /* input & positions */
                for (int j = 0; j <= biasUnit; j++) {
                    dW[i][j] += errorSignalMemoryState[u][v] * sc[u][v][j];
                }
            }
            i++;
        }
    }

    /**
     * Computes derivatives
     *
     * @param element
     */
    public void derivatives(int element) {

        for (int u = 0; u < numBlocks; u++) {

            for (int v = 0; v < blockSize[u]; v++) {
                float tmp1 = activationInputGate[u] * g[u][v];
                float tmp2 = (1.0f - activationInputGate[u]) * tmp1;
                float tmp3 = (1.0f - 0.5f * g[u][v]) * tmp1;
                for (int j = 0; j < numInputsLocalCoding; j++) {
                    /* weights to input gate */
                    //si[u][v][inputIndex[element][j]] += tmp2 * inputVector[element][j];
                    /* weights to cell input */
                    sc[u][v][inputIndex[element][j]] += tmp3; // * inputVector[element][j];
                }

                /* position inputs */
                for (int j = 0; j < numGaussians; j++) {
                    sc[u][v][inputIndex[element][numInputsLocalCoding + j]] += tmp3 * inputVector[element][numInputsLocalCoding + j];
                }
                sc[u][v][biasUnit] += tmp3;
                si[u][v][biasUnit] += tmp2;

                for (int j = biasUnit + 1; j <= lastMemoryUnit; j++) {
                    /* weights to input gate */
                    si[u][v][j] += tmp2 * activationOld[j];
                    /* weights to cell input */
                    sc[u][v][j] += tmp3 * activationOld[j];
                }
            }
        }
    }


    /**
     * Initialization of the net
     *
     * @param bias_inp
     * @param bias_out
     * @param bias_memin
     * @param weight_memout
     * @param outputbias
     */
    public void init(float[] bias_inp, float[] bias_out,
                     float[] bias_memin, float[] weight_memout,
                     float outputbias) {
        cloneWeightMatrix();
    }

    public void fillDWMatrix(int seqNr) {
        for (int i = 0; i < numUnits; i++) {
            System.arraycopy(dW[i], 0, dwMatrix[seqNr][i], 0, numUnits);
        }
    }

    public void resetCrossEntropyErrors() {
        for (int i = 0; i < numSeqsTrain; i++) {
            crossEntropyErrorsTrain[i] = 0;
        }
        for (int i = 0; i < numSeqsTest; i++) {
            crossEntropyErrorsTest[i] = 0;
        }
    }


    /**
     * Reset the net
     *
     */
    public void reset() {

        for (int i = 0; i < numUnits; i++) {
            activationNew[i] = 0.5f;
            activationOld[i] = 0.5f;
            for (int j = 0; j < numUnits; j++) {
                dW[i][j] = 0;
            }
        }

        activationNew[biasUnit] = 1.0f;
        activationOld[biasUnit] = 1.0f;

        for (int u = 0; u < numBlocks; u++) {
            activationInputGate[u] = 0;
            activationOutputGate[u] = 0;
            for (int v = 0; v < blockSize[u]; v++) {
                memoryState[u][v] = 0;
                g[u][v] = 0;
                h[u][v] = 0;
                activationCell[u][v] = 0.5f;
                for (int j = 0; j <= lastMemoryUnit; j++) {
                    si[u][v][j] = 0;
                    sc[u][v][j] = 0;
                }
            }
        }
    }

    public void setInputVector(float[][] inputVector) {
        this.inputVector = inputVector;
    }

    public void setInputIndex(short[][] inputIndex) {
        this.inputIndex = inputIndex;
    }

    public void setTargets(float[] targets) {
        this.targets = targets;
    }

    public void setEpoch_err(float epoch_err) {
        this.epochError = epoch_err;
    }

    public void setFalsepos(int falsepos) {
        this.falsepos = falsepos;
    }

    public void setFalseneg(int falseneg) {
        this.falseneg = falseneg;
    }

    public float[][] getDW() {
        return dW;
    }

    public float[][][] getDwMatrix() {
        return dwMatrix;
    }

    public float[] getCrossEntropyErrorsTrain() {
        return crossEntropyErrorsTrain;
    }

    public float[] getCrossEntropyErrorsTest() {
        return crossEntropyErrorsTest;
    }

}
