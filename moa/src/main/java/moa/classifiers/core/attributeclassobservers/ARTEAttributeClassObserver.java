package moa.classifiers.core.attributeclassobservers;

import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.GaussianEstimator;

public class ARTEAttributeClassObserver extends GaussianNumericAttributeClassObserver{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	protected Random rand; 
    protected int count = 0;

    public void setRand(Random rand) {
    	this.rand = rand; 
    }
    
    public Random getRand() {
    	return rand;
    }
	
	@Override
    public AttributeSplitSuggestion getBestEvaluatedSplitSuggestion(
            SplitCriterion criterion, double[] preSplitDist, int attIndex,
            boolean binaryOnly) {
        
    	
    	AttributeSplitSuggestion bestSuggestion = null;
        double[] suggestedSplitValues = getSplitPointSuggestions();
       
        for (double splitValue : suggestedSplitValues) {
            double[][] postSplitDists = getClassDistsResultingFromBinarySplit(splitValue);
            double merit = criterion.getMeritOfSplit(preSplitDist,
                    postSplitDists);
            if ((bestSuggestion == null) || (merit > bestSuggestion.merit)) {
                bestSuggestion = new AttributeSplitSuggestion(
                        new NumericAttributeBinaryTest(attIndex, splitValue,
                        true), postSplitDists, merit);
            }
        }
       
        return bestSuggestion;
    }

    public double[] getSplitPointSuggestions() {
    	Set<Double> suggestedSplitValues = new TreeSet<Double>();
        double minValue = Double.POSITIVE_INFINITY;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.attValDistPerClass.size(); i++) {
            GaussianEstimator estimator = this.attValDistPerClass.get(i);
            if (estimator != null) {
                if (this.minValueObservedPerClass.getValue(i) < minValue) {
                    minValue = this.minValueObservedPerClass.getValue(i);
                }
                if (this.maxValueObservedPerClass.getValue(i) > maxValue) {
                    maxValue = this.maxValueObservedPerClass.getValue(i);
                }
            }
        }
        if (minValue < Double.POSITIVE_INFINITY) {
         
        	double splitValue = (rand.nextDouble() * (maxValue - minValue)) + minValue;
        	if ((splitValue > minValue) && (splitValue < maxValue)) {
                suggestedSplitValues.add(splitValue);
            }
        	
        }
        double[] suggestions = new double[suggestedSplitValues.size()];
        int i = 0;
        for (double suggestion : suggestedSplitValues) {
            suggestions[i++] = suggestion;
        }
        return suggestions;
    }

}
