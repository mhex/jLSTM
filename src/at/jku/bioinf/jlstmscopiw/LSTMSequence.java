/**
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
 * $Id: LSTMSequence.java 95 2008-10-07 06:25:00Z mhe $
 *
 */

package at.jku.bioinf.jlstmscopiw;

/**
 * Represents a sequence with an array of local coding indices
 * and the label.
 * 
 * @author mhe
 *
 */
public class LSTMSequence {
    
    short[] localCodings;
    float[] targets;
    boolean toTrain; // flag indicating if this sequence has to be trained (error still to large)
    
    float[][] positionCodings;
    int[][] positionIndices;
    
    public LSTMSequence(short[] localCodings, float[][] positionCodings, int[][] positionIndices, float[] targets) {
        
        this.localCodings = localCodings;
        this.targets = targets;
        this.toTrain = true;
        
        this.positionCodings = positionCodings;
        this.positionIndices = positionIndices;
        
    }

    public short[] getLocalCodings() {
        return localCodings;
    }
    
    public float[][] getPositionCodings() {
        return positionCodings;
    }

    public int[][] getPositionIndices() {
        return positionIndices;
    }

    public float[] getTargets() {
        return targets;
    }
    
    public boolean getToTrain() {
    	return toTrain;
    }
    
    public void setToTrain(boolean flag) {
    	this.toTrain = flag;
    }
    
    public LSTMSequence clone() {
    	return(new LSTMSequence(this.localCodings, this.positionCodings, this.positionIndices, this.targets));
    }
        

}
