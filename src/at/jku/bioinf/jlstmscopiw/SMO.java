package at.jku.bioinf.jlstmscopiw;

import java.util.Arrays;


/* Calculating alphas for importance weights
 * 
 *  
 *  
 *  
 *  */
public class SMO {
    
    double[][] dwMatrix;
	
	public SMO(int numExamples, int dimdWTo, int dimdWFrom) {
		
	    dwMatrix = new double[numExamples][dimdWTo * dimdWFrom];
		
	}
	
	/* Calculate the alphas
	 * 
	 * @param dE: Matrix of Gradients of the loss function with respect to parameters
     *         dim(dE) = [ num_weights, num_samples  ]
     * @param p: Free parameter > 0
     * @param E: E absolut erreors of missclassification
     * @param C: upper bound for alpha
	 * 
	 * 
	 * @param Q weight derivatives
	 */
	public float[] solve(float[][][] dwMatrices, float[] c, float l, float u, int threadNr) {
	    
	    
	    /* Parameters */
	    double EE    = 0; // objective is zero
        //double tol = 0.1;
        double tolab = 0.001;
        double maxUp = 5.0 * tolab;
        double avUp  = maxUp;
        //double stepp = 0;
        double EEo   = EE;
	    
        //double minImprov = 1e-3;
        
        double epsZero = 0.0000001;

        /* Flatten dw matrix */
		
	    int numExamples = c.length;
        
        int dimDWTo = dwMatrices[0].length;
        int dimDWFrom = dwMatrices[0][0].length;
        
        System.out.println("Thread " + threadNr + " SMO: W " + numExamples + " " + dimDWTo + " " + dimDWFrom);
        System.out.println("Thread " + threadNr + " SMO: flatten matrix..");
        
        double min = Integer.MAX_VALUE;
        double max = Integer.MIN_VALUE;
    
        double sum = 0;
        
        for (int i = 0; i < numExamples; i++) {
            int index = 0;
            for (int j = 0; j < dimDWTo; j++) {
                for (int k = 0; k < dimDWFrom; k++) {
                    dwMatrix[i][index] = dwMatrices[i][j][k];
                    min = Math.min(dwMatrix[i][index], min);
                    max = Math.max(dwMatrix[i][index], max);
                    sum += dwMatrix[i][index];
                    index++;
                }
            }
        }       

        System.out.format("SMO: Thread %d dwMat: min: %.6f max: %.6f mean %.6f", threadNr, min, max, sum/(numExamples * dimDWTo * dimDWFrom));
        System.out.println();
    
        /* Init variables */
	    double[] alphas = new double[numExamples];
	    Arrays.fill(alphas,  0);
	    double[] alphasOld = new double[numExamples];
	    //double[] alphasDiffs = new double[numExamples];
	    double[][] K = new double[numExamples][numExamples];
        double[] uvec = new double[numExamples];
        Arrays.fill(uvec,  u);
        double[] lvec = new double[numExamples];
        Arrays.fill(lvec, l);
        boolean[] notSet = new boolean[numExamples];
        Arrays.fill(notSet, true);
        double[] F = new double[numExamples];
        double eps = 1.0 - epsZero;
        double[] alphaStat = new double[numExamples];
        double[] scaler = new double[numExamples];
        for (int i = 0; i < numExamples; i++) {
            double s = 0;
            for (int j = 0; j < numExamples; j++) {
                s += dwMatrix[i][j] * dwMatrix[i][j];
            }
            s = Math.sqrt(s + 0.0001);
            s = 1.0;
            F[i] = -(c[i] / s);
            uvec[i] *= s;
            scaler[i] = s;
            K[i][i] = 1.0;
        }

	    int indAlpha1 = 0;
	    int indAlpha2 = 0;
	    int steps = 0;
	    double rma1 = 0;
        double rma2 = 0;
        
        /* Find alphas */
	    while (avUp > tolab) { // && numAlphasNotChanged < maxAlphasNotChanged) {
	        maxUp = -1.0;
	        for (int i = 0; i < numExamples; i++) {
	            double fi = F[i];
	            if (fi < 0 && alphas[i] < uvec[i] - epsZero) {
	                fi = -fi;
	                double fTest = uvec[i] - alphas[i];
	                if (fi > fTest) fi = fTest;
	                if (fi > maxUp) {
	                    maxUp = fi;
	                    indAlpha1 = i;
	                }
	            } else {
	                if (fi > 0 && alphas[i] > lvec[i] + epsZero) {
	                    double fTest = alphas[i] - lvec[i];
	                    if (fi > fTest) fi = fTest;
	                    if (fi > maxUp) {
	                        maxUp = fi;
	                        indAlpha1 = i;
	                    }
	                }
	            }
	        }

	        double f1 = F[indAlpha1];
	        double a1 = alphas[indAlpha1];
	        double l1 = lvec[indAlpha1];
	        double u1 = uvec[indAlpha1];
	        
	        // Calculate kernel entry for indAlpha1
	        if (notSet[indAlpha1]) {
	            notSet[indAlpha1] = false;
	            for (int i = 0; i < numExamples; i++) {
	                if (notSet[i]) {
	                    double kij = 0;
	                    for (int j = 0; j < numExamples; j++) {
	                        kij += dwMatrix[indAlpha1][j] * dwMatrix[i][j];
	                    }
	                    kij /= scaler[indAlpha1] * scaler[i];
	                    K[indAlpha1][i] = kij;
	                    K[i][indAlpha1] = kij;
	                }
	            }
	        }
	        
	        // Now search for alpha_2 for alpha_1
	        
	        maxUp = 0;
	        
	        for (int i = 0; i < numExamples; i++) {
	            double k12 = K[indAlpha1][i]; // 0 <= k12 < 1
	            if (k12 > eps) k12 = eps;
	            double tmp = 1.0 - k12 * k12;
	            double f2 = F[i];
	            double optUnboundA1 = (f2 * k12 - f1) / tmp;
	            double optUnboundA2 = (f1 * k12 - f2) / tmp;
	            
	            // check if updates are on bound
	            // -1 -> not on bound
	            //  0 -> on lower bound
	            //  1 -> on upper bound
	            double b1 = -1;  // bound information for alpha1
	            double b2 = -1;  // bound information for alpha2

	            // binary bound indicator: 0 -> not on bound, 1 -> on bound
	            double onbound1 = 0; // bound indicator of alpha1
	            double onbound2 = 0; // bound indicator of alpha2

	            // violations of constraints for updates are checked
	            // and corrected for alpha1 and alpha2
	            if (optUnboundA1 + a1 < l1) {
	                optUnboundA1 = l1 - a1;
	                onbound1 = 1;
	                b1 = 0;
	            } else if (optUnboundA1 + a1 > u1) {
	                optUnboundA1 = u1 - a1;
	                onbound1 = 1;
	                b1 = 1;
	            }

	            if (optUnboundA2 + alphas[i] < lvec[i]) {
	                optUnboundA2 = lvec[i] - alphas[i];
	                onbound2 = 1;
	                b2 = 0;
	            } else if (optUnboundA2 + alphas[i] > uvec[i]) {
	                optUnboundA2 = uvec[i] - alphas[i];
	                onbound2 = 1;
	                b2 = 1;
	            }

	            // bidk (bound indikator) indicates if:
	            // no alpha is on bound     = 0
	            // only alpha1 is on bound  = 1
	            // only alpha2 is on bound  = 2
	            // both alphas are on bound = 3
	           double  bidk = 2 * onbound2 + onbound1;
	            //System.out.println("DEBUG:::bound_indikator = " + bidk);
	            if (bidk > 0) {
	                if (bidk == 1) {
	                    optUnboundA2 = -(k12 * optUnboundA1 + f2);
	                    
	                    if (optUnboundA2 + alphas[i] < lvec[i]) {
	                        optUnboundA2 = lvec[i] - alphas[i];
	                        onbound2 = 1;
	                        //bidk = 3;
	                    } else if (optUnboundA2 + alphas[i] > uvec[i]) {
	                        optUnboundA2 = uvec[i] - alphas[i];
	                        onbound2 = 1;
	                        //bidk = 3;
	                    }
	                    
	                } 
	                if (bidk == 2) {
	                    optUnboundA1 = -(k12 * optUnboundA2 + f1);
	                    
	                    if (optUnboundA2 + a1 < l1) {
	                        optUnboundA1 = l1 - a1;
	                        onbound1 = 1;
	                        //bidk = 3;
	                    } else if (optUnboundA1 + a1 > u1) {
	                        optUnboundA1 = u1 - a1;
	                        onbound1 = 1;
	                        //bidk = 3;
	                    }
	                    //*/
	                }
	                if (bidk == 3) {
	                    // corner indicates, if:
	                    // on lower/lower corner  => 0
	                    // on upper/lower corner  => 1
	                    // on lower/upper corner  => 2
	                    // on upper/upper corner  => 3
	                    double corner = 2 * b2 + b1;
	                    //System.out.println("DEBUG:::corner = " + corner);
	                    if (corner == 0) {
	                        double g2 = optUnboundA2 + optUnboundA1 * k12 + f2;
	                        if (g2 < 0) {
	                            optUnboundA2 = -(k12 * optUnboundA1 + f2);
	                            if (optUnboundA2 + alphas[i] < lvec[i]) {
	                                optUnboundA2 = lvec[i] - alphas[i];
	                            } else if (optUnboundA2 + alphas[i] > uvec[i]) {
	                                optUnboundA2 = uvec[i] - alphas[i];
	                            }
	                        } else {
	                            double g1 = optUnboundA1 + optUnboundA2 * k12 + f1;
	                            if (g1 < 0) {
	                                optUnboundA1 = -(k12 * optUnboundA2 + f1);
	                                if (optUnboundA1 + a1 < l1) {
	                                    optUnboundA1 = l1 - a1;
	                                } else if (optUnboundA1 + a1 > u1) {
	                                    optUnboundA1 = u1 - a1;
	                                }
	                            }
	                        }
	                    }
	                    if (corner == 1) {
	                        double g2 = optUnboundA2 + optUnboundA1 * k12 + f2;
	                        if (g2 < 0) {
	                            optUnboundA2 = -(k12 * optUnboundA1 + f2);
	                            if (optUnboundA2 + alphas[i] < lvec[i]) {
	                                optUnboundA2 = lvec[i] - alphas[i];
	                            } else if (optUnboundA2 + alphas[i] > uvec[i]) {
	                                optUnboundA2 = uvec[i] - alphas[i];
	                            }
	                        } else {
	                            double g1 = optUnboundA1 + optUnboundA2 * k12 + f1;
	                            if (g1 > 0) {
	                                optUnboundA1 = -(k12 * optUnboundA2 + f1);
	                                if (optUnboundA1 + a1 < l1) {
	                                    optUnboundA1 = l1 - a1;
	                                } else if (optUnboundA1 + a1 > u1) {
	                                    optUnboundA1 = u1 - a1;
	                                }
	                            }
	                        }
	                    }
	                    if (corner == 2) {
	                        double g2 = optUnboundA2 + optUnboundA1 * k12 + f2;
	                        if (g2 > 0) {
	                            optUnboundA2 = -(k12*optUnboundA1 + f2);
	                            if (optUnboundA2 + alphas[i] < lvec[i]) {
	                                optUnboundA2 = lvec[i] - alphas[i];
	                            } else if (optUnboundA2 + alphas[i] > uvec[i]){
	                                optUnboundA2 = uvec[i] - alphas[i];
	                            }
	                        } else {
	                            double g1 = optUnboundA1 + optUnboundA2 * k12 + f1;
	                            if (g1 < 0) {
	                                optUnboundA1 = -(k12 * optUnboundA2 + f1);
	                                if (optUnboundA1 + a1 < l1) {
	                                    optUnboundA1 = l1 - a1;
	                                } else if (optUnboundA1 + a1 > u1) {
	                                    optUnboundA1 = u1 - a1;
	                                }
	                            }
	                        }
	                    }
	                    if (corner == 3) {
	                        double g2 = optUnboundA2 + optUnboundA1 * k12 + f2;
	                        if (g2 > 0) {
	                            optUnboundA2 = -(k12*optUnboundA1 + f2);
	                            if (optUnboundA2 + alphas[i] < lvec[i]) {
	                                optUnboundA2 = lvec[i] - alphas[i];
	                            } else if (optUnboundA2 + alphas[i] > uvec[i]) {
	                                optUnboundA2 = uvec[i] - alphas[i];
	                            }
	                        } else {
	                            double g1 = optUnboundA1 + optUnboundA2 * k12 + f1;
	                            if (g1 > 0) {
	                                optUnboundA1 = -(k12 * optUnboundA2 + f1);
	                                if (optUnboundA1 + a1 < l1) {
	                                    optUnboundA1 = l1 - a1;
	                                } else if (optUnboundA1 + a1 > u1) {
	                                    optUnboundA1 = u1 - a1;
	                                }
	                            }
	                        }
	                    }
	                }
	            }
	            
	            double mxa = Math.abs(optUnboundA1);
	            
	            if (mxa <  Math.abs(optUnboundA2)) {
	                mxa = Math.abs(optUnboundA2);
	            }

	            if (mxa > maxUp) {
	                maxUp = mxa;
	                rma1 = optUnboundA1;
	                rma2 = optUnboundA2;
	                indAlpha2 = i;
	            }

	        }

	        // calculate kernel entry for found indAlphas2
	        if (notSet[indAlpha2]) {

	            notSet[indAlpha2] = false;

	            for (int i = 0; i < numExamples; i++) {

	                if (notSet[i]) {   // calculate only examples not already calculated

	                    double kij = 0;

	                    for (int j = 0; j < numExamples; j++) {

	                        kij += dwMatrix[indAlpha2][j] * dwMatrix[i][j];

	                    }

	                    kij /= scaler[indAlpha2] * scaler[i];

	                    K[indAlpha2][i] = kij;

	                    K[i][indAlpha2] = kij;

	                }
	            }
	        }
	        
	        double k12 = K[indAlpha1][indAlpha2];
	        if (k12 > eps) {
	            k12 = eps;
	        }
	        double f2 = F[indAlpha2];
	        double a2 = alphas[indAlpha2];
	        double l2 = lvec[indAlpha2];
	        double u2 = uvec[indAlpha2];

	        if (rma1 + a1 > u1) {
	            rma1 = u1 - a1;
	        }

	        if (rma1 + a1 < l1) {
	            rma1 = l1 - a1;
	        }

	        if (rma2 + a2 > u2) {
	            rma2 = u2 - a2;
	        }

	        if (rma2 + a2 < l2) {
	            rma2 = l2 - a2;
	        }

	        alphas[indAlpha1] += rma1;
	        alphas[indAlpha2] += rma2;

	        // update objective
	        EE += rma1 * (0.5 * rma1 + f1 + rma2 * k12) + rma2 * (0.5 * rma2 + f2);
	        
	        /*
	        System.out.format("a1 %.4f f1: %.4f l1: %.4f u1: %.4f", a1, f1, l1, u1);
	        System.out.println();
	        System.out.format("a2 %.4f f2: %.4f l2: %.4f u2: %.4f", a2, f2, l2, u2);
	        System.out.println();
	        System.out.format("rma1 %.4f rma2: %.4f EE: %.4f", rma1, rma2, EE);
	        System.out.println();
	        //print "DEBUG:::EE = " + str(EE)
            */
	        // updates F
	        for (int i = 0; i < numExamples; i++) {
	            F[i] += rma1 * K[i][indAlpha1] + rma2 * K[i][indAlpha2];
	        }

	        steps++;
	        
	        double improvement = Math.abs(EE-EEo) / (Math.abs(EEo) + tolab);
	        //System.out.println("DEBUG:::improvement = " + improvement);
	        EEo = EE;
	        avUp = 0.8 * avUp + 0.2 * improvement;

	        /*
	        boolean[] comp = compareAlphas(alphasOld, alphas, epsZero);
	        
	        if (allTrue(comp)) {
	            numAlphasNotChanged += 1;
	        } else {
	            numAlphasNotChanged = 0;
	            boolean[] changed = where(comp, false);
	            for (int i = 0; i < alphaStat.length; i++) {
	                if (changed[i]) {
	                    alphaStat[i]++;
	                }
	            }
	        }
	        */
	        
	    }
        
	    float[] alphasFloat = new float[numExamples];
	    
	    for (int i = 0; i < numExamples; i++) {
	        
	        alphasFloat[i] = (float)(alphas[i] / scaler[i]);
	        
	    }
	    
	    return(alphasFloat);
		
		
	}
	
	
	private boolean[] compareAlphas(double[] a1, double[] a2, double eps) {
	    
	    boolean[] res = new boolean[a1.length];
	    
	    for (int i = 0; i < a1.length; i++) {
	    
	        res[i] = (Math.abs(a1[i] - a2[i]) < eps) ? true : false;
	        
	    }
	    //ret = arr.all()
	    //print "DEBUG:::ret = " + str(ret)
	    
	    return(res);
	}
	
	private boolean allTrue(boolean[] bVec) {
	    
	    for (int i = 0; i < bVec.length; i++) {
	        
	        if (!bVec[i]) return false;
	        
	    }
	    
	    return(true);
	    
	}
	
	private boolean[] where(boolean[] bVec, boolean val) {
	    
	    boolean[] res = new boolean[bVec.length];
	    
	    for (int i = 0; i < bVec.length; i++) {
	        
	        res[i] = (bVec[i] == val) ? true : false;
	        
	    }
	    
	    return(res);
	    
	}

}
