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
 * $Id: WeightMatrix.java 232 2009-02-16 07:30:13Z mhe $
 *
 */

package at.jku.bioinf.jlstmscopiw;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.InputMismatchException;
import java.util.Random;
import java.util.Scanner;

/**
 * Represents a weight matrix with some methods.
 *
 * @author mhe
 *
 */
public class WeightMatrix {

    float[][] weightMatrix;

    int numUnits;
    int lastInputUnit;
    int biasUnit;
    int lastMemoryUnit;
    int firstOutputUnit;
    int numBlocks;
    int nSymbols;

    int[] blockSize;

    float alpha;
    float alphaInput;

    int wmdimx;
    int wmdimy;
    int wmdimyoffset;

    final float[][] regMat;

    float[][] adaG;
    float[][] adaS;

    String weightfile;

    public WeightMatrix (int numUnits, int lastMemoryUnit, int lastInputUnit, int biasUnit, int numBlocks,
            float[] biasInputGate, float[] biasOutputGate, float[] biasMemoryInput,
            float[] weightToOutput, float biasOutput,
            int[] blockSize, int nSymbols, float initRange, float alpha) {

        final int randfactor = 1000000;

        this.numUnits    = numUnits;
        this.lastMemoryUnit   = lastMemoryUnit;
        this.lastInputUnit     = lastInputUnit;
        this.firstOutputUnit = lastMemoryUnit + 1;
        this.biasUnit   = biasUnit;
        this.numBlocks = numBlocks;
        this.blockSize = blockSize;
        this.nSymbols   = nSymbols;

        this.alpha = alpha;
        this.alphaInput = alpha * 0.00000001f;

        weightMatrix = new float[numUnits][numUnits];

        wmdimx       = numUnits;
        wmdimyoffset = biasUnit + 1;
        wmdimy       = numUnits - wmdimyoffset;
        
        regMat = RegularizationMatrices.getGonnet500skal0();

        //dwOld = new float[numUnits][numUnits];
        adaG = new float[numUnits][numUnits];
        adaS = new float[numUnits][numUnits];

        // Init weights
        for (int i = 0; i < numUnits; i++) {

            for (int j = 0; j <= lastInputUnit; j++) {

                weightMatrix[i][j] = 0;

            	if (i > lastInputUnit + 1 && i < firstOutputUnit) {

		  if ((i - lastInputUnit - 1) % 3 == 0) {

            		weightMatrix[i][j] = seprand(randfactor * 2) - randfactor;
            		weightMatrix[i][j] /= randfactor;
            		weightMatrix[i][j] *= initRange;

            	  }

            	}

            }

            for (int j = biasUnit; j < firstOutputUnit; j++) {

               /*
            	weightMatrix[i][j] = seprand(randfactor * 2) - randfactor;
        		weightMatrix[i][j] /= randfactor;
        		weightMatrix[i][j] *= initRange;
                */
                weightMatrix[i][j] = 0;

            }

            for (int j = firstOutputUnit; j < numUnits; j++) {

            	weightMatrix[i][j] = 0;

            }

        }

        /* bias and weight to output unit initalizations */

        int i = biasUnit + 1;

        for (int u = 0; u < numBlocks; u++) {

            /* bias to input gate */
            weightMatrix[i][biasUnit] = biasInputGate[u];

            i++;

            /* bias to output gate */
            weightMatrix[i][biasUnit] = biasOutputGate[u];

            for (int v = 0; v < blockSize[u]; v++) {

            	i++;

            }
            /* bias to memory cell */
            weightMatrix[i][biasUnit] = biasMemoryInput[u];

            /* weight to output unit */
            for (int j = firstOutputUnit; j < numUnits; j++) {

            	//if ((j % 2) == 0) {
            		weightMatrix[j][i] = weightToOutput[u];
            	//} else {
            	//	weightMatrix[j][i] = weightToOutput[u];
            	//}

            }

            i++;

        }

        /* output bias */
        for (i = firstOutputUnit; i < numUnits; i++) {

        	weightMatrix[i][biasUnit] = biasOutput;

        }


    }

    /**
     * Reads a weight matrix from file system
     *
     */
    public void readWeightMatrix() {

    	/* reset weight matrix to zero */
    	for (int i = 0; i < this.wmdimx; i++) {
    		for (int j = 0; j < this.wmdimx; j++) {
    			weightMatrix[i][j] = 0;
    		}
    	}

    	Scanner s = null;

    	try {

            s = new Scanner(new BufferedReader( new FileReader( weightfile ) ));

            String dim = s.next("dim");

            if (!"dim".equals(dim)) {

            	System.err.println("Error in weight matrix. Wrong dimension header");
            	System.exit(100);

            }

            this.wmdimy = s.nextInt();
            this.wmdimx = s.nextInt();
            this.wmdimyoffset = this.wmdimx - this.wmdimy;

            for ( int i = 0; i < wmdimy; i++ ) {

                int j;

                int k = wmdimyoffset + i;

                for (j = 0; j < wmdimx; j++) {

                    weightMatrix[k][j] = s.nextFloat();

                }

            }

            s.close();

          }

          catch ( FileNotFoundException fnfe ) {
        	  System.err.println( "Weight matrix file not found" );
        	  System.exit(101);
          }

          catch (InputMismatchException ime ) {

        	  System.err.println( "Error in weight matrix, input mismatch" );
        	  System.exit(102);

          }

          finally {

        	  s.close();

          }

    }

    /**
     * writes the current weight matrix to a file
     *
     */
    public void writeWeightMatrix() {

        BufferedWriter out = null;

        try {

          out = new BufferedWriter( new FileWriter( weightfile ) );

          out.write("dim " + String.valueOf(wmdimy) + " " + String.valueOf(wmdimx) + "\n");

          for ( int i = 0; i < wmdimy; i++ ) {

              int j;

              int k = wmdimyoffset + i;

              for (j = 0; j < wmdimx - 1; j++) {

                  out.write(new DecimalFormat("#.########").format(weightMatrix[k][j]) + " ");

              }

              out.write(new DecimalFormat("#.########").format(weightMatrix[k][j]));

              out.write("\n");

          }

          out.write("\n");

          out.close();

        }

        catch ( IOException e ) {
          System.err.println( e );
        }

        finally {
          try {
            if ( out != null ) out.close();
          } catch ( IOException e ) { }
        }

    }



    /**
     * write a better readable weight matrix
     *
     */
    public void writeReadableWeightMatrix() {

        BufferedWriter out = null;

        String rweightfile = "readable_" + weightfile;

        try {

          out = new BufferedWriter( new FileWriter( rweightfile ) );

          out.write("dim " + String.valueOf(wmdimy) + ":" + String.valueOf(wmdimx));
          out.newLine();

          for ( int i = 0; i < wmdimy; i++ ) {
              int j;
              int k = wmdimyoffset + i;
              for (j = 0; j < wmdimx - 1; j++) {
                  out.write("(" + String.valueOf(k) + "," + String.valueOf(j) + "): ");
                  out.write(new DecimalFormat("#.####").format(weightMatrix[k][j]) + " ");
              }
              out.write("(" + String.valueOf(k) + "," + String.valueOf(j) + "): ");
              out.write(new DecimalFormat("#.####").format(weightMatrix[k][j]));
              out.newLine();
          }
          out.newLine();
          out.close();
        }
        catch ( IOException e ) {
          System.err.println( e );
        }
        finally {
          try {
            if ( out != null ) out.close();
          } catch ( IOException e ) { }
        }

    }


    /**
     * Computes weight updates
     *
     * A regularization is done on the input weights with the Gonnet500 matrix
     *
     * @param dW
     */
    public void weightUpdateIW(float[][] dW, int threadNr) {
        
        System.out.println("Thread " + threadNr + " Updating weights..");

        /* output units */

        for (int k = firstOutputUnit; k < numUnits; k++) {

            /* memory cell */

            int i = biasUnit + 1;

            for (int u = 0; u < numBlocks; u++) {

                i++;

                for (int v = 0; v < blockSize[u]; v++) {

                    i++;

                    weightMatrix[k][i] +=  alpha * dW[k][i];

                }

                i++;

            }

            weightMatrix[k][biasUnit] +=  dW[k][biasUnit];

        }

        /* memory cells with gates */

        int i = biasUnit + 1;

        for (int u = 0; u < numBlocks; u++) {

        	/* weights to input gate */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {           

                weightMatrix[i][j] +=  alpha * dW[i][j];

            }

            i++;

            /* weights to output gate */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {
                
                weightMatrix[i][j] += alpha * dW[i][j];

            }

            /* weights to memory cell */
            for (int v = 0; v < blockSize[u]; v++) {

                i++;

                for (int j = 0; j <= lastInputUnit; j++) {
                
                    weightMatrix[i][j] +=  alpha * dW[i][j];
	    				
                }
                

                /* Positions and bias */
                for (int j = lastInputUnit + 1; j <= biasUnit; j++) {

                    weightMatrix[i][j] += alpha * dW[i][j];

                }

                /* Recurrent connections from other memory cells */
                /*
                for (int j = biasUnit;j <= lastMemoryUnit; j++) {

                	dW[i][j] = momFact * dwOld[i][j] + dW[i][j];

                    weightMatrix[i][j] += dW[i][j];

                    dwOld[i][j] = dW[i][j];

                }
                */

            }

            i++;

        }


    }
    
    /**
     * Computes weight updates
     *
     * A regularization is done on the input weights with the Gonnet500 matrix
     *
     * @param dW
     */
    public void weightUpdate(float[][] dW) {
        
        float[] dwReg = new float[nSymbols];

        //final float wd = 0.999999;

        /* output units */

        for (int k = firstOutputUnit; k < numUnits; k++) {

            /* memory cell */

            int i = biasUnit + 1;

            for (int u = 0; u < numBlocks; u++) {

                i++;

                for (int v = 0; v < blockSize[u]; v++) {

                    i++;

                    /*
                    adaG[k][i] = (1- decay) *  dW[k][i] * dW[k][i] + decay * adaG[k][i];

                    dW[k][i] = Math.sqrt(adaS[k][i] + offset) / Math.sqrt(adaG[k][i] + offset) * dW[k][i];

                    adaS[k][i] = (1 - decay) * dW[k][i] * dW[k][i] + decay * adaS[k][i];
                    */

                    weightMatrix[k][i] +=  alpha * dW[k][i];

                    //weightMatrix[k][i] +=  dW[k][i];

                    //weightMatrix[k][i] = (weightMatrix[k][i] > 1) ? 1 : weightMatrix[k][i];
                    //weightMatrix[k][i] = (weightMatrix[k][i] < -1) ? -1 : weightMatrix[k][i];

                }

                i++;

            }

            /* bias */

            /*
            adaG[k][biasUnit] = (1- decay) *  dW[k][biasUnit] * dW[k][biasUnit] + decay * adaG[k][biasUnit];

            dW[k][biasUnit] = Math.sqrt(adaS[k][biasUnit] + offset) / Math.sqrt(adaG[k][biasUnit] + offset) * dW[k][biasUnit];

            adaS[k][biasUnit] = (1 - decay) * dW[k][biasUnit] * dW[k][biasUnit] + decay * adaS[k][biasUnit];
            */

            weightMatrix[k][biasUnit] +=  alpha * dW[k][biasUnit];

            //weightMatrix[k][biasUnit] += dW[k][biasUnit];

            //weightMatrix[k][biasUnit] = (weightMatrix[k][biasUnit] > 1) ? 1 : weightMatrix[k][biasUnit];
            //weightMatrix[k][biasUnit] = (weightMatrix[k][biasUnit] < -1) ? -1 : weightMatrix[k][biasUnit];


        }

        /* memory cells with gates */

        int i = biasUnit + 1;

        for (int u = 0; u < numBlocks; u++) {

            /* weights to input gate */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {

                /*
                adaG[i][j] = (1- decay) *  dW[i][j] * dW[i][j] + decay * adaG[i][j];

                dW[i][j] = Math.sqrt(adaS[i][j] + offset) / Math.sqrt(adaG[i][j] + offset) * dW[i][j];

                adaS[i][j] = (1 - decay) * dW[i][j] * dW[i][j] + decay * adaS[i][j];
                */

                weightMatrix[i][j] +=  alpha * dW[i][j];

            }

            i++;

            /* weights to output gate */
            for (int j = biasUnit; j <= lastMemoryUnit; j++) {

                /*
                adaG[i][j] = (1- decay) *  dW[i][j] * dW[i][j] + decay * adaG[i][j];

                dW[i][j] = Math.sqrt(adaS[i][j] + offset) / Math.sqrt(adaG[i][j] + offset) * dW[i][j];

                adaS[i][j] = (1 - decay) * dW[i][j] * dW[i][j] + decay * adaS[i][j];
                */

                weightMatrix[i][j] += alpha * dW[i][j];

            }

            /* weights to memory cell */
            for (int v = 0; v < blockSize[u]; v++) {

                i++;

                /* Regularization with the Gonnet500 substitution matrix */


                int segment = 0; // begin of current AA segment (vector of nSymbols, local coding for one AA)
                int l = 0; // index of current AA in current AA segment (0:(nSymobls-1))


                for (int k = 0; k < nSymbols; k++) {

                    dwReg[k] = 0;

                }


                for (int j = 0; j <= lastInputUnit; j++, l++) {
                
                    /* end of segment reached, now weight update with aggregated updates */
                    if (l == nSymbols) {

                        for (int k = 0; k < nSymbols; k++) {

                            weightMatrix[i][segment + k] +=  alpha * dwReg[k];

                            dwReg[k] = 0;

                        }

                        segment += nSymbols;

                        l = 0;

                    }

                    // Aggregate updates of all weights in the current segment,
                    // current update weighted by the regularization matrix vector e.g.. Gonnet
                    // The vector of current AA is used.
                    for (int k = 0; k < nSymbols; k++) {

                        // Gonnet500 matrix gives good results
                        if (regMat[l][k] != 0) {

                            dwReg[k] += dW[i][j] * regMat[l][k];

                        }

                    }


                }
                

                /* Positions and bias */
                for (int j = lastInputUnit + 1; j <= biasUnit; j++) {

                    weightMatrix[i][j] += alpha * dW[i][j];

                }

                /* Recurrent connections from other memory cells */
                /*
                for (int j = biasUnit;j <= lastMemoryUnit; j++) {

                    dW[i][j] = momFact * dwOld[i][j] + dW[i][j];

                    weightMatrix[i][j] += dW[i][j];

                    dwOld[i][j] = dW[i][j];

                }
                */

            }

            i++;

        }


    }
   

    /**
     * Calculate final batch dW with QP solutions alphas
     */
    public float[][] calcBatchdW(float[][][] dwMatrixExamples, float[] alphas, float dwCut, int threadNr) {
        
        int numExamples = dwMatrixExamples.length;
        
        int dimDWto = dwMatrixExamples[0].length;
        
        int dimDWfrom = dwMatrixExamples[0][0].length;
        
        System.out.println("Thread " + threadNr + " Calculating batch dW " + numExamples + " " + dimDWto + " " + dimDWfrom);
        
        float[][] dW = new float[dimDWto][dimDWfrom];
        
        for (int i = 0; i < numExamples; i++) {
            
            for (int j = 0 ; j < dimDWto; j++) {
                
                for (int k = 0; k < dimDWfrom; k++) {
                    
                   dW[j][k] += dwMatrixExamples[i][j][k] * alphas[i];
                   
                }
            }
            
        }
   
        float min = 10000000;
        float max = -10000000;
        float sumdW = 0;
        
        for (int j = 0 ; j < dimDWto; j++) {
            
            for (int k = 0; k < dimDWfrom; k++) {
                
                min = (dW[j][k] < min) ? dW[j][k] : min;
        
                max = (dW[j][k] > max) ? dW[j][k] : max;
                
                sumdW += Math.abs(dW[j][k]);
                        
            }
            
        }
        
        System.out.format("Thread %d batchDWs: min: %.6f max: %.6f absmean %.6f", threadNr, min, max, sumdW/(dimDWto * dimDWfrom));
        System.out.println();
        
        min = 10000000;
        max = -10000000;
        sumdW = 0;
        
        for (int j = 0 ; j < dimDWto; j++) {
            
            for (int k = 0; k < dimDWfrom; k++) {
                
               dW[j][k] =  (dW[j][k] > dwCut) ? dwCut : dW[j][k];
               dW[j][k] =  (dW[j][k] < -dwCut) ? -dwCut : dW[j][k];
               
               min = (dW[j][k] < min) ? dW[j][k] : min;
               max = (dW[j][k] > max) ? dW[j][k] : max;
               sumdW += Math.abs(dW[j][k]);
               
            }
        }
        
        System.out.format("Thread %d batchDWsCorr: min: %.6f max: %.6f absmean %.6f", threadNr, min, max, sumdW/(dimDWto * dimDWfrom));
        System.out.println();
        
        
        return(dW);
        
    }

    /**
     * set the weight file name
     *
     * @param weightfile
     */
    public void setWeightfile(String weightfile) {
        this.weightfile = weightfile;
    }

    /**
     * get the weight matrix
     *
     * @return weightMatrix
     */
    public float[][] getWeightMatrix() {
        return weightMatrix;
    }

    /**
     * Computes a random integer
     *
     * @param k
     * @return
     */
    int seprand(int k) {
        long l;
        int f;
        Random rand = new Random();
        l = rand.nextLong();
        f = (int)l % k;
        return(f);
      }


}
