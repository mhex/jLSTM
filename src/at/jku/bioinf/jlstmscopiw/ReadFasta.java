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
 *	 title = {{Fast Model-based Protein Homology Detection without Alignment}},
 *	 journal = {Bioinformatics},
 *	 volume = {23},
 *	 number = {14},
 *	 pages = {1728-1736},
 *	 doi = {doi:10.1093/bioinformatics/btm247},
 *	 year = {2007},
 *	 URL = {http://bioinformatics.oxfordjournals.org/cgi/content/abstract/btm247v1},
 *	 eprint = {http://bioinformatics.oxfordjournals.org/cgi/reprint/btm247v1}
 * }
 *
 * $Id: ReadFasta.java 236 2009-02-17 18:32:40Z mhe $
 *
 */

package at.jku.bioinf.jlstmscopiw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.NoSuchElementException;
import java.util.Scanner;

import org.biojava.bio.BioException;
import org.biojavax.bio.seq.RichSequence;
import org.biojavax.bio.seq.RichSequenceIterator;
import org.biojavax.bio.seq.RichSequence.IOTools;


/**
 * Reads a FASTA file and computes indices for sparse local coding.
 *
 * @author mhe
 */
public class ReadFasta {

	int numPositionBins;
	int numGaussians;

    ArrayList<GaussStruct> positionVector;

    float[][] positionCodings;
    int[][] positionIndices;

    public ReadFasta(int nSymbols, int numposbins, int numgauss) {

		this.numPositionBins = numposbins;
		this.numGaussians   = numgauss;

		this.positionVector = new ArrayList<GaussStruct>(numposbins);

		for (int i = 0; i < numposbins; i++) {

			GaussStruct gs = new GaussStruct();
			gs.gauss = 0;
			gs.idx   = 0;
			positionVector.add(gs);

		}

    }

    /*
     * Data structure that holds a gauss value and an
     * index for an input vector.
     */
    class GaussStruct implements Comparable<Object> {

    	float gauss;
    	int idx;

    	public int compareTo(Object obj) {

    	    GaussStruct gs = (GaussStruct) obj;

    	    if ( gauss == gs.gauss ) {
    			return 0;
    		}
    		else if ( gauss > gs.gauss) {
    			return -1;
    		}
    		else return 1;

    	}

    }

    public class GaussStructComparator implements Comparator<Object> {

    	public int compare(Object obj1, Object obj2) {

    		GaussStruct gs1 = (GaussStruct) obj1;
    	    GaussStruct gs2 = (GaussStruct) obj2;

    	    int gsComp = gs1.compareTo(gs2);

    	    return ((gsComp == 0) ? gs1.compareTo(gs2) : gsComp);

    	  }

    }

    float lstmgauss(float m, float s, float x) {
		return (float) ((40.0 / s) * Math.exp(-0.5 * Math.pow((x - m) / (0.5 * s),2)));
	}


    private void computePositionCodings( RichSequence rs ) {

    	positionCodings = new float[rs.length()][numGaussians];
    	positionIndices = new int[rs.length()][numGaussians];

    	if (numPositionBins > 0) {

    		float sd = (float)rs.length() / (numPositionBins - 1.0f);

    		for (int i = 0; i < rs.length(); i++) {
    			for (int j = 0; j < numPositionBins; j++) {
    				this.positionVector.get(j).gauss = lstmgauss((float)j * sd, sd, (float)i);
    				this.positionVector.get(j).idx = j;
    			}
    			Collections.sort(this.positionVector, new GaussStructComparator());
    			positionCodings[i][numGaussians / 2] = positionVector.get(0).gauss;
    			positionIndices[i][numGaussians / 2] = positionVector.get(0).idx;
    			for (int j = 1, k = 1; j < numGaussians / 2 + 1; j++, k += 2) {
    				//System.out.println(j + " " + k + " " + numGaussians / 2);
    				positionCodings[i][numGaussians / 2 - j] = positionVector.get(k).gauss;
    				positionIndices[i][numGaussians / 2 - j] = positionVector.get(k).idx;
    				positionCodings[i][numGaussians / 2 + j] = positionVector.get(k + 1).gauss;
    				positionIndices[i][numGaussians / 2 + j] = positionVector.get(k + 1).idx;
    				//System.out.printf("%d %f %d %f %d %f %d\n", i, positionCodings[i][numGaussians / 2 - j], positionIndices[i][numGaussians / 2 - j], positionCodings[i][numGaussians / 2],  positionIndices[i][numGaussians / 2], positionCodings[i][numGaussians / 2 + j], positionIndices[i][numGaussians / 2 + j]);
    			}
    		}

    	}

    }


    /*
     * Returns the position of given char in aaChars array
     *
     */
    private byte getAACharPos(char aa) {

        /*
          ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR
          TRP TYR VAL ASX GLX X
        */

        final char aaChars[] =
            {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
             'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X'};


        int i;
        for (i = 0; i < 23; i++) {
            if (aaChars[i] == aa) {
                return (byte)i;
            }
        }
        if (aa == 'x' || aa == 'U') {
            return 22;
        }

        /* TODO throw Exception */
        System.out.println("Error in sequencegeneration, unknown AA Symbol");
        System.out.println("AA Symbol: " + aa);
        System.exit(-1);

        return 0;

    }


    /**
     * Reads a FASTA file with help of Biojava RichSequence.IOTools.
     * Creates a LSTMSequence for any sequence
     * and label and stores it in an ArrayList which is returned.
     *
     * @param filename
     * @param label
     * @return ArrayList<LSTMSequence>
     */
    public ArrayList<LSTMSequence> read(String fastaFilename, String labelsFilename, int numOutputs) {

    	Scanner s = null;

    	try {
    		s = new Scanner(new File(labelsFilename));
    	} catch (FileNotFoundException fnfex) {
            //problem reading file
            System.err.println("jLSTM: Could not open label file: " + labelsFilename);
            System.exit(1);
        }

    	ArrayList<Integer> labels = new ArrayList<Integer>();
    	while (s.hasNext()){
    		int label = Integer.valueOf(s.next());
    		if (label < 1) {
    			System.err.println("found a label 0 or negative.");
    			System.exit(1);
    		}
    		if (label > numOutputs) {
    			System.err.println("found a label greater than number of outputs.");
    			System.exit(1);
    		}
    	    labels.add(label);
    	}
    	s.close();

    	float[][] targets = new float[labels.size()][numOutputs];

    	for (int i = 0; i < labels.size(); i++) {

    		for (int j = 0; j < numOutputs; j++) {

    			targets[i][j] = 0.0f;

    		}

    		targets[i][labels.get(i) - 1] = 1.0f;

    	}

        ArrayList<LSTMSequence> seqs = new ArrayList<LSTMSequence>();

        try {
            // Setup file input
            BufferedReader br =
                new BufferedReader(new FileReader(fastaFilename));

            // RichSequenceIterator
            RichSequenceIterator rsi = IOTools.readFastaProtein(br, null);

            int i = 0;

            // For each sequence do
            while(rsi.hasNext()) {

            	if (i + 1 > labels.size()) {

            		System.err.println("Error: Number of sequences " + (i + 1) + " is greater than number of labels " + labels.size() +".");

            		System.exit(1);

            	}

                try {

                    RichSequence rs = rsi.nextRichSequence();

                    //System.out.println(rs.getName() + " " + rs.getDescription());

                    // Sequence data
                    char seq[] = new char[rs.length()];
                    short localCodings[] = new short[rs.length()];

                    rs.seqString().getChars(0, rs.length(), seq, 0);

                    // Compute local coding indices
                    for (int j = 0; j < rs.length(); j++) {
                        localCodings[j] = getAACharPos(seq[j]);
                    }

                    computePositionCodings(rs);

                    // Store codings and label in a LSTMSequence
                    LSTMSequence lstmseq = new LSTMSequence(localCodings, positionCodings.clone(), positionIndices.clone(), targets[i]);

                    // Add LSTMSequence to ArrayList
                    seqs.add(lstmseq);

                }
                catch (BioException ex) {
                    //not in fasta format or wrong alphabet
                    System.out.println("not in fasta format or wrong alphabet");
                    System.exit(0);
                }
                catch (NoSuchElementException ex) {
                    //no fasta sequences in the file
                    System.out.println("no fasta sequences in the file");
                    System.exit(0);
                }

                i++;

            }

            br.close();

            if (i  != labels.size()) {

        		System.err.println("Error: Number of labels is greater than the number of sequences.");

        		System.exit(1);

        	}


        }
        catch (FileNotFoundException fnfex) {
            //problem reading file
            System.err.println("jLSTM: Could not open sequence file: " + fastaFilename);
            System.exit(1);
        }

        catch (IOException ioe) {
        	  //problem reading file
            System.err.println("jLSTM: Could not close sequence file: " + fastaFilename);
            //System.exit(1);
        }

        return seqs;

    }
}
