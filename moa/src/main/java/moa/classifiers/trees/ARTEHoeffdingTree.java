package moa.classifiers.trees;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.ARTEAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.conditionaltests.NominalAttributeMultiwayTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Utils;

/**
 * Adaptive Random Tree Ensemble - Hoeffding Tree. 
 * 
 * <p>This is the base model for the  Adaptive Random Forest ensemble learner. 
 * This Hoeffding Tree includes a subspace size random k parameter, which defines the number of randomly  
 * selected features to be considered at each split with cut point random. </p>
 * 
 * @author Aldo
 * @version $Revision: 1 $
 */
public class ARTEHoeffdingTree extends HoeffdingTree {
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k',
            "Number of features per subset for each node split. Negative values = #features - k", 
            2, Integer.MIN_VALUE, Integer.MAX_VALUE);
    
    
    
    @Override
    public String getPurposeString() {
        return "Adaptive Random Tree Ensemble - Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomTreeEnsemble.";
    }
    
    
    public static class RandomLearningNode extends ActiveLearningNode {
        
        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;
        
        // The split attribute
        protected int m_attIndex;

        // The split point
        protected double m_splitPoint;
        
        
        public RandomLearningNode(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations);
            this.numAttributes = subspaceSize;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {            
        	this.observedClassDistribution.addToValue((int) inst.classValue(),
                    inst.weight());
            if (this.listAttributes == null) {
                this.listAttributes = new int[this.numAttributes];
                for (int j = 0; j < this.numAttributes; j++) {
                    boolean isUnique = false;
                    while (isUnique == false) {

                    	this.listAttributes[j] = ht.classifierRandom.nextInt(inst.numAttributes() - 1);
                        isUnique = true;
                        for (int i = 0; i < j; i++) {
                            if (this.listAttributes[j] == this.listAttributes[i]) {
                                isUnique = false;
                                break;
                            }
                        }
                    }

                }
            }
            
            for (int j = 0; j < this.numAttributes; j++) {
                int i = this.listAttributes[j];
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    
                	if (inst.attribute(instAttIndex).isNominal())
                		obs = ht.newNominalClassObserver();
                	else {
                		obs = ht.newNumericClassObserver();
                		
                		ARTEAttributeClassObserver aetg = (ARTEAttributeClassObserver)obs;
                		if (aetg.getRand() == null) {
                            aetg.setRand(ht.classifierRandom);
                		}
                	}
                	
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }
        }
        
        @Override
        public AttributeSplitSuggestion[] getBestSplitSuggestions(
                SplitCriterion criterion, HoeffdingTree ht) {
        	
        	List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
        	double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                // add null split as an option
                bestSuggestions.add(new AttributeSplitSuggestion(null,
                        new double[0][], criterion.getMeritOfSplit(
                        preSplitDist,
                        new double[][]{preSplitDist})));
            }
            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                	AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }
        
        
    }

    public static class LearningNodeNB extends RandomLearningNode {

        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
                return NaiveBayes.doNaiveBayesPrediction(inst,
                        this.observedClassDistribution,
                        this.attributeObservers);
            }
            return super.getClassVotes(inst, ht);
        }

        @Override
        public void disableAttribute(int attIndex) {
            // should not disable poor atts - they are used in NB calc
        }
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {

        private static final long serialVersionUID = 1L;

        protected double mcCorrectWeight = 0.0;

        protected double nbCorrectWeight = 0.0;
        
        public LearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            int trueClass = (int) inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }
            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }
            super.learnFromInstance(inst, ht);
        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            if (this.mcCorrectWeight > this.nbCorrectWeight) {
                return this.observedClassDistribution.getArrayCopy();
            }
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution, this.attributeObservers);
        }
    }

    public ARTEHoeffdingTree() {
        this.removePoorAttsOption = null;
    }
    
    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new RandomLearningNode(initialClassObservations, this.subspaceSizeOption.getValue());
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations, this.subspaceSizeOption.getValue());
        } else { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations, this.subspaceSizeOption.getValue());
        }
        return ret;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
    	
    	
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNode)) {
                ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                
                if (weightSeen
                        - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
                	attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
        
        if (this.trainingWeightSeenByModel
                % this.memoryEstimatePeriodOption.getValue() == 0) {
            estimateModelByteSizes();
        }
    }
    
    @Override
    protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,
            int parentIndex) {
    	if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) {
                    shouldSplit = true;
                }
                // }
                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2 - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }
            if (shouldSplit) {
            	
            	AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                
            	if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);
                } else {
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(),splitDecision.numSplits() );
                    
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                    	if (splitDecision.splitTest instanceof NominalAttributeMultiwayTest) {
                    		if (this.classifierRandom.nextDouble() > 0.5) {
                        		continue;
                        	}
                    	}
                    	
                    	Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                    	newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }
            	
            	// manage memory
                enforceTrackerLimit();
            }
        }
    }
    
}
