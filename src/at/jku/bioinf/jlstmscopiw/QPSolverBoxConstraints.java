package at.jku.bioinf.jlstmscopiw;

//import java.util.Arrays;


import java.util.Arrays;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.optimizers.JOptimizer;
import com.joptimizer.optimizers.OptimizationRequest;

import Jama.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.*;
import jcuda.runtime.JCuda;

/* Calculating alphas for importance weights */
public class QPSolverBoxConstraints {
	
	float[] dwMatrix;
	float[][] dwMatrixA;
	Matrix dwMatrixM;
	float[] kernelVector;
	Matrix dwK;
	Matrix uvecTemplate, lvecTemplate, pTemplate;
	int numExamples;
	int gpu;
	double[][] I, Ineg;
	
	public QPSolverBoxConstraints(int numExamples, int dimdWTo, int dimdWFrom, float l, float u, int gpu) {

	    dwMatrix = new float[numExamples * dimdWTo * dimdWFrom];
	    //dwMatrixM = new Matrix(numExamples, dimdWTo * dimdWFrom);
	    kernelVector = new float[numExamples * numExamples];
	    dwK = new Matrix(numExamples, numExamples);
		uvecTemplate = new Matrix(numExamples, 1, u);
		lvecTemplate = new Matrix(numExamples, 1, l);
		pTemplate    = new Matrix(numExamples, 1, 0);
		
        // Identity matrix
	    I = new double[numExamples][numExamples];
	    for(int i = 0; i < numExamples; i++) I[i][i] = 1.0;
	    
	    // Negative identity matrix
	    Ineg = new double[numExamples][numExamples];
        for(int i = 0; i < numExamples; i++) Ineg[i][i] = -1.0;
		
		this.numExamples = numExamples;
		this.gpu     = gpu;
		
	}
	
	public float[] solveJO(Matrix Q, float[] cFloat, float l, float u, int threadNr) {

	    double[] uvec = uvecTemplate.copy().getColumnPackedCopy();
        double[] lvec = lvecTemplate.copy().getColumnPackedCopy();
        // Cast to double
        double[] c = new double[cFloat.length];
        for (int i = 0 ; i < cFloat.length; i++) {
            c[i] = -cFloat[i];
            //uvec[i] = -uvec[i];
        }
	    // Objective function
        PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(Q.getArray(), c, 0);

        //inequalities
        ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2 * numExamples];
        for (int i = 0; i < numExamples; i++) {
            inequalities[i] = new LinearMultivariateRealFunction(Ineg[i], lvec[i]);
        }
        for (int i = 0; i < numExamples; i++) {
            inequalities[numExamples + i] = new LinearMultivariateRealFunction(I[i], -uvec[i]);
        }

        /* Doesn't work
        ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2];
        inequalities[0] = new PDQuadraticMultivariateRealFunction(Ineg, lvec, 0);
        inequalities[1] = new PDQuadraticMultivariateRealFunction(Ineg, uvec, 0);
        */
        
        //optimization problem
        OptimizationRequest or = new OptimizationRequest();
        or.setF0(objectiveFunction);
        //or.setInitialPoint(new double[] { 0.1, 0.9});
        or.setFi(inequalities);
        or.setToleranceFeas(1.E-5);
        or.setTolerance(1.E-5);
        //optimizer
        JOptimizer opt = new JOptimizer();
        opt.setOptimizationRequest(or);
        //optimization
        try {
            int returnCode = opt.optimize();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            System.exit(-1);
        }
	    
        //get results
        double[] sol = opt.getOptimizationResponse().getSolution();
        float[] solFloat = new float[numExamples];
        for (int i = 0 ; i < numExamples; i++) {
            solFloat[i] = (float) sol[i];
        }
        return(solFloat);
	}
	
		
	/* Calculate the alphas
	 * @param Q weight derivatives kernel matrix */
	public float[] solve(Matrix H, float[] cFloat, float l, float u, int threadNr) {
	    
	    System.out.println("Thread " + threadNr + " QP solving..");
	   
	    // Cast to double
	    double[] c = new double[cFloat.length];
	    for (int i = 0 ; i < cFloat.length; i++) {
	        c[i] = cFloat[i];
	    }
	    Matrix uvec = uvecTemplate.copy();
	    Matrix lvec = lvecTemplate.copy();
	    Matrix p = pTemplate.copy();
	    Matrix x = uvec.plus(lvec).times(0.5);
	    Matrix Hx = H.times(x);
	    Matrix xHx = x.transpose().times(Hx);
	    Matrix cM = new Matrix(c, numExamples);
	    Matrix cx = cM.transpose().times(x);
	    Matrix Qx = xHx.times(0.5).plus(cx);
	    Matrix g = Hx.plus(cM);
	    Matrix QxOld = Qx.copy();
	    boolean[] AOld = logicalOr(x, l, u);
	    Matrix gR = g.copy();

	    for (int i = 0; i < gR.getRowDimension(); i++) {
            if (AOld[i]) gR.set(i, 0, 0);
        }
	    
	    Matrix gP = g.copy();      
	            
	    double sigma = 0.1;
	    double gamma = 0.9;
	    double eta = g.norm2();
	    double gR2 = gR.norm2();
	    double gR2Old = gR2;
	    
	    Matrix xold = x.plus(new Matrix(numExamples, 1, 0.01));
	    Matrix xnew;
	    
	    boolean[] A, B;
	    double gp, alpha, beta;
	    int cycleCounter = 0;
	    int maxCycles = 10000;
	    
	    while (true) {
	        if (cycleCounter++ > maxCycles) {
	            System.out.format("Thread %d QP solver break: cylce counter > %d gR2: %.4f", threadNr, maxCycles, gR2);
	            System.out.println();
	            break;
	        }
	        if (gR2 < 0.01 || logicalAllMatrix(x, xold)) {
	            System.out.format("Thread %d QP solver break: gR2 %.4f x=xold: %s cycleCounter: %d", threadNr, gR2, logicalAllMatrix(x, xold), cycleCounter);
	            System.out.println();
	            break;
	        }
	        A = logicalOr(x, l, u);
	        B = logicalOrElementwise(logicalAnd(x, l, g, true), logicalAnd(x, u, g, false));
	        gR = g.copy();
	        for (int i = 0; i < gR.getRowDimension(); i++) {
	            if (A[i]) gR.set(i, 0, 0);
	        }
	        gP = g.copy();
	        for (int i = 0; i < gP.getRowDimension(); i++) {
	            if (B[i]) gP.set(i, 0, 0);
	        }
	        gR2 = gR.norm2();
	        if (Math.sqrt(gR2) <= eta) {
	            if (logicalAll(AOld, B)) {
	                beta = gR2 / gR2Old;
	            } else {
	                beta = 0;
	            }
	            p = gP.plus(p.times(beta)).times(-1);
	        } else {
	            if (logicalAll(AOld,A)) {
	                beta = gR2 / gR2Old;
	            } else {
	                beta = 0;
	            }
	            p = gR.plus(p.times(beta)).times(-1);;
	        }
	        if (logicalAllMatScalar(p, 0)) {
	            System.out.println("Thread " + threadNr + " QP solver break: all ps = 0 cyclecounter: " + cycleCounter);
	            break;
	        }

	        gp = g.transpose().times(p).get(0,0);
	        alpha = - gp / p.transpose().times(H).times(p).get(0, 0);
	        while (true) {
	            xnew = x.plus(p.times(alpha));
	            xnew = minElementwise(uvec.getRowPackedCopy(),
	                    maxElementwise(lvec.getRowPackedCopy(),
	                            xnew.getRowPackedCopy()));
	            Hx = H.times(xnew);
	            xHx = xnew.transpose().times(Hx);
	            cx = cM.transpose().times(xnew);
	            Qx = xHx.times(0.5).plus(cx);

	            if (Qx.minus(QxOld).get(0, 0) <= gamma * alpha * gp) {
	                break;
	            }
	            alpha *= sigma;
	        }
	        xold = x.copy();
	        x = xnew.copy();
	        g = Hx.plus(cM);
	        gR2Old = gR2;
	        QxOld = Qx.copy();
            System.arraycopy(A, 0, AOld, 0, A.length);
	        eta *= 0.5;
	    }

	    float[] xFloat = new float[numExamples];
	    for (int i = 0 ; i < numExamples; i++) {
	        xFloat[i] = (float) x.get(i, 0);
	    }

	    return(xFloat);		

	}
	
    Matrix calcDWKernelGPU(float[][][] dwMatrices, int threadNr) {
        
        int numExamples = dwMatrices.length;
        int dimdWTo = dwMatrices[0].length;
        int dimdWFrom = dwMatrices[0][0].length;
        int dwVectorLength = dimdWTo * dimdWFrom;
        int dwMatSize = numExamples * dwVectorLength;
        int kMatSize = numExamples * numExamples;
        
        System.out.println("Thread " + threadNr + " Kernel GPU: " + numExamples + " " + dimdWTo + " " + dimdWFrom);
        System.out.println("Thread " + threadNr + " flatten matrix..");
        
        int index = 0;

        for (float[][] matrix : dwMatrices) {
            for (int j = 0; j < dimdWTo; j++) {
                for (int k = 0; k < dimdWFrom; k++) {
                    dwMatrix[index] = matrix[j][k];
                    index++;
                }
            }
        }
        
        // Initialize JCublas
        
        int error;
        
        error = JCublas.cublasInit();
        if (error != 0) {
            System.out.println("Cublas init error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasShutdown();
            System.exit(error);
        }
       
        //error = JCuda.cudaSetDevice(deviceNr - 1);
        error = JCuda.cudaSetDevice(gpu);
        if (error != 0) {
            System.out.println("Set device error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasShutdown();
            System.exit(error);
        }
                
        System.out.println("GPU " + gpu + " CUDA alloc..");
        
        // Allocate memory on the device
        Pointer dwMP = new Pointer();
        Pointer dwKernelP = new Pointer();
        
        error = JCublas.cublasAlloc(dwMatSize, Sizeof.FLOAT, dwMP);
        if (error != 0) {
            System.out.println("Cublas alloc error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasShutdown();
            System.exit(error);
        }
        
        error = JCublas.cublasAlloc(kMatSize , Sizeof.FLOAT, dwKernelP);
        if (error != 0) {
            System.out.println("Cublas alloc error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasFree(dwMP);
            JCublas.cublasShutdown();
            System.exit(error);
        }        
        
        System.out.println("GPU " + gpu + " CUDA copy..");
        
        // Copy the memory from the host to the device
        error = JCublas.cublasSetVector(dwMatSize, Sizeof.FLOAT, Pointer.to(dwMatrix), 1, dwMP, 1);
        if (error != 0) {
            System.out.println("Cublas copy vector error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasFree(dwMP);
            JCublas.cublasFree(dwKernelP);
            JCublas.cublasShutdown();
            System.exit(error);
        }
        
        error = JCublas.cublasSetVector(kMatSize, Sizeof.FLOAT, Pointer.to(kernelVector), 1, dwKernelP, 1);
        if (error != 0) {
            System.out.println("Cublas copy vector error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasFree(dwMP);
            JCublas.cublasFree(dwKernelP);
            JCublas.cublasShutdown();
            System.exit(error);
        }
        
        System.out.println("GPU " + gpu + " CUDA sgemm..");

        float alpha = 1;
        float beta = 0;

        // Execute sgemm
        JCublas.cublasSgemm('t', 'n',
                numExamples, numExamples, dwVectorLength,
                alpha, dwMP, dwVectorLength,
                dwMP, dwVectorLength, beta,
                dwKernelP, numExamples);

        System.out.println("GPU " + gpu + " CUDA get results..");

        error = JCublas.cublasGetVector(kMatSize, Sizeof.FLOAT, dwKernelP, 1, Pointer.to(kernelVector), 1);
        if (error != 0) {
            System.out.println("Cublas get vector error: " + error + " " + JCuda.cudaGetErrorString(error));
            JCublas.cublasFree(dwMP);
            JCublas.cublasFree(dwKernelP);
            JCublas.cublasShutdown();
            System.exit(error);
        }
        
        System.out.println("GPU " + gpu + " CUDA clean up..");
        
        // Clean up
        JCublas.cublasFree(dwMP);
        JCublas.cublasFree(dwKernelP);

        JCublas.cublasShutdown();
        
        System.out.println("Thread " + threadNr + " fill mat..");

        double max = Integer.MIN_VALUE;
        double min = Integer.MAX_VALUE;
        double sumK = 0;
        index = 0;

        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < numExamples; j++) {
                dwK.set(i,j, kernelVector[index]);
                max = (kernelVector[index] > max) ? kernelVector[index] : max;
                min = (kernelVector[index] < min) ? kernelVector[index] : min;
                sumK += kernelVector[index];
                index++;
            }
        }

        System.out.format("Thread %d GPU Kernel: %d,%d min: %.4f max: %.4f mean %.4f", threadNr, dwK.getRowDimension(), dwK.getColumnDimension(), min, max, sumK / kMatSize);
        System.out.println();       
        
        return(dwK);
        
    }
	
	
	Matrix calcDWKernelCPU(float[][][] dwMatrices, int threadNr) {
	    
        numExamples = dwMatrices.length;
        int dimDWTo = dwMatrices[0].length;
        int dimDWFrom = dwMatrices[0][0].length;
        int kMatSize = numExamples * numExamples; 
        
        System.out.println("Thread " + threadNr + " Kernel CPU: " + numExamples + " " + dimDWTo + " " + dimDWFrom);
        System.out.println("Thread " + threadNr + " flatten matrix..");
        
        for (int i = 0; i < numExamples; i++) {
            int index = 0;
            for (int j = 0; j < dimDWTo; j++) {
                for (int k = 0; k < dimDWFrom; k++) {
                    dwMatrixM.set(i,index, dwMatrices[i][j][k]);
                    index++;
                }
            }
        }       
        
        System.out.println("Thread " + threadNr + " Calculating kernel.. ");
        
        Matrix dwK = dwMatrixM.times(dwMatrixM.transpose());
        double max = Integer.MIN_VALUE;
        double min = Integer.MAX_VALUE;
        float sumK = 0;
        
        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < numExamples; j++) {
                    max = (dwK.get(i,j) > max) ? dwK.get(i,j) : max;
                    min = (dwK.get(i,j) < min) ? dwK.get(i,j) : min;
                    sumK += dwK.get(i,j);
            }
        }
        
        System.out.format("Thread %d CPU Kernel: %d,%d min: %.4f max: %.4f mean %.4f", threadNr, numExamples, numExamples, min, max, sumK / kMatSize);
        System.out.println();
        
        return(dwK);
        
	}
	
	boolean[] logicalOr(Matrix x, float l, float u) {
	    
	    int vLength = x.getRowDimension();
	    boolean[] resultVector = new boolean[vLength];

	    for (int i = 0; i < vLength; i++) {
	        if (x.get(i,  0) == l || x.get(i, 0) == u) {
	            resultVector[i] = true;
	        }
	    }
	    
	    return resultVector;
	    
	}
	
    boolean[] logicalAnd(Matrix x, float val, Matrix g, boolean gt) {
        
        int vLength = x.getRowDimension();
        boolean[] resultVector = new boolean[vLength];
        
        if (gt) {
            for (int i = 0; i < vLength; i++) {
                if (x.get(i,  0) == val && g.get(i, 0) >= 0) {
                    resultVector[i] = true;
                }
            }
        } else {
            for (int i = 0; i < vLength; i++) {
                if (x.get(i,  0) == val && g.get(i, 0) <= 0) {
                    resultVector[i] = true;
                }
            }
        }
        
        return resultVector;
        
    }
    
    boolean[] logicalOrElementwise(boolean[] a, boolean[] b) {
        
        boolean[] resultVector = new boolean[a.length];
        
        for (int i = 0; i < a.length; i++) {
            resultVector[i] = a[i] && b[i];
        }
        
        return(resultVector);
        
    }
	
    boolean logicalAll(boolean[] a, boolean[] b) {
                
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) return(false);
        }
        
        return(true);
        
    }
    
    boolean logicalAllMatScalar(Matrix a, int val) {
        
        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                if (a.get(i,j) != val) return(false);
            }
        }
        
        return(true);
        
    }
    
    boolean logicalAllMatrix(Matrix a, Matrix b) {
        
        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                if (a.get(i,j) != b.get(i,j)) return(false);
            }
        }
        
        return(true);
        
    }
    
    double[] maxElementwise(double[] a, double[] b) {

	    double[] resultVector = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            resultVector[i] = Math.max(a[i], b[i]);
        }
        
        return(resultVector);
        
    }
    
    Matrix minElementwise(double[] a, double[] b) {
        
        double[] resultVector = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            resultVector[i] = (a[i] < b[i]) ? a[i] : b[i];
        }
        
        return(new Matrix(resultVector, a.length));
        
    }
    
    /**
     * 
     * @param a
     *            A(m,n)
     * @return vector representation of A(m,n):
     *         (A(0,0)...A(0,n))...(A(m,0)...A(m,n))
     */
    float[] To1D(float[][] A) {
        int m = A.length;
        int n = A[0].length;
        float[] v = new float[m * n];
        int count = 0;
        for (float[] floats : A) {
            for (int j = 0; j < n; j++) {
                v[count] = floats[j];
                count += 1;
            }
        }
        return v;
    }
 
    /**
     * 
     * @param v
     *            vector representation of A(m,n)
     *            (A(0,0)...A(0,n))...(A(m,0)...A(m,n))
     * @param m
     * @param n
     * @return A(m,n)
     */
    float[][] To2D(float[] v, int m, int n) {
        float[][] A = new float[m][n];
        int count = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = v[count];
                count += 1;
            }
        }
        return A;
    }    

}
