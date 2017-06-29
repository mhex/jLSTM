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
 * $Id: InputMatrix.java 222 2009-02-12 15:29:27Z mhe $
 *
 */

package at.jku.bioinf.jlstmscopiw;

/**
 * Holds the input data for LSTM network. For a sequence the
 * complete input matrix is created.
 * 
 * @author mhe
 *
 */
public class InputMatrix {
    
    float[][]  inputVector;
    short[][] inputIndex;
      
    /**
     * Creates the input matrix from a LSTMSequence
     * 
     * @see LSTMSequence
     * 
     * @param lstmSequence
     * @param windowSize  
     * @param nSymbols
     */
    public void computeInputs(LSTMSequence lstmSequence, int windowSize, int nSymbols, int numGaussians) {
        
        /* half of the window */
        int wside = (windowSize - 1) / 2;
        
        int seqLength = lstmSequence.getLocalCodings().length;
        
        float[][] positionCodings = lstmSequence.getPositionCodings();
        int[][] positionIndices    = lstmSequence.getPositionIndices();
        
        /* Input vector (only 1s for local coding) */
        inputVector = new float[seqLength][windowSize + numGaussians];
        /* Index for local coding */
        inputIndex  = new short[seqLength][windowSize + numGaussians];
        
        int numNAInputs   =  nSymbols * windowSize;

        /*
        
        Order of AAs
        
        ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR 
        TRP TYR VAL ASX X GLX
        
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
        'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X'
        
        */
        
        /* Values for local coding */
        
        final float[] aaVal = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 0.0f };

        
        short[] localCodings = lstmSequence.getLocalCodings();
        
        for (int i = 0; i < seqLength; i++) {
                 
            int starti = i - wside;

            /* amino coding */
            for (int j = 0; j < windowSize; j++) {
                int iter = starti + j;
                if (iter < 0 || iter >= seqLength) {
                    inputVector[i][j] = 0;
                    inputIndex[i][j] = 0;          
                }
                else {
                    int aaCharPos = localCodings[iter];
                    inputVector[i][j] = aaVal[aaCharPos];
                    inputIndex[i][j] = (short)(j * nSymbols + aaCharPos);
                }
            }
            
            for (int j = 0; j < numGaussians; j++) {
            	
            	inputVector[i][j + windowSize] = positionCodings[i][j];
            	inputIndex[i][j + windowSize]  = (short)(numNAInputs + positionIndices[i][j]);
            	
            }
            
        }
            
    }
    
    public float[][] getInputVector() {
        return inputVector;
    }


    public short[][] getInputIndex() {
        return inputIndex;
    }

}
