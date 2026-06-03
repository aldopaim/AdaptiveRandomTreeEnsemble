package moa.classifiers.core.attributeclassobservers;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeRandomBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.DoubleVector;
import moa.core.Utils;
import java.util.Random;

public class ARTENominalAttributeClassObserver extends NominalAttributeClassObserver {

    private static final long serialVersionUID = 1L;
    private Random random; 

    
    public void setRand(Random rand) {
    	this.random = rand; 
    	
    }
    
    public Random getRand() {
    	return this.random;
    }
    
    @Override
    public AttributeSplitSuggestion getBestEvaluatedSplitSuggestion(
            SplitCriterion criterion,
            double[] preSplitDist,
            int attIndex,
            boolean binaryOnly) {

        
        if (!binaryOnly) {
            return null;
        }

        int numValues = getTotalObservedValues();
        if (numValues < 2) {
            return null;
        }

        
        boolean[] mask = generateRandomBinaryMask();

        double[][] postSplit = getClassDistsResultingFromArbitraryBinarySplit(mask);

        double weightA = Utils.sum(postSplit[0]);
        double weightB = Utils.sum(postSplit[1]);
        double total = weightA + weightB;

        
        if (total == 0 || weightA < 0.01 * total || weightB < 0.01 * total) {
            return null;
        }

        double merit = criterion.getMeritOfSplit(preSplitDist, postSplit);
        InstanceConditionalTest test = new NominalAttributeRandomBinaryTest(attIndex, mask);

        return new AttributeSplitSuggestion(test, postSplit, merit);
    }

    
    private boolean[] generateRandomBinaryMask() {
        int max = getMaxNumValues();
        
        
        if (max < 2) {
            return new boolean[0]; 
        }
        
        boolean[] mask = new boolean[max];

        int split = 1 + random.nextInt(max - 1); 
        for (int i = 0; i < split; i++) mask[i] = true;

        
        for (int i = 0; i < max; i++) {
            int j = random.nextInt(max);
            boolean temp = mask[i];
            mask[i] = mask[j];
            mask[j] = temp;
        }

        return mask;
    }

    
    public double[][] getClassDistsResultingFromArbitraryBinarySplit(boolean[] subSetAMask) {
        DoubleVector distA = new DoubleVector();
        DoubleVector distB = new DoubleVector();

        for (int c = 0; c < this.attValDistPerClass.size(); c++) {
            DoubleVector attValDist = this.attValDistPerClass.get(c);
            if (attValDist != null) {
                for (int v = 0; v < attValDist.numValues(); v++) {
                    double w = attValDist.getValue(v);
                    if (v < subSetAMask.length && subSetAMask[v]) {
                        distA.addToValue(c, w);
                    } else {
                        distB.addToValue(c, w);
                    }
                }
            }
        }
        return new double[][]{distA.getArrayRef(), distB.getArrayRef()};
    }

    
    private int getTotalObservedValues() {
        int count = 0;
        for (DoubleVector dv : this.attValDistPerClass) {
            if (dv != null) {
                for (int i = 0; i < dv.numValues(); i++) {
                    if (dv.getValue(i) > 0) count++;
                }
            }
        }
        return count;
    }

    private int getMaxNumValues() {
        int max = 0;
        for (DoubleVector dv : this.attValDistPerClass) {
            if (dv != null && dv.numValues() > max) max = dv.numValues();
        }
        return max;
    }
}