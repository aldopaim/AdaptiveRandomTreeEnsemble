package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.AbstractMOAObject;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.trees.ARTEHoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;

/**
 * Adaptive Random Tree Ensemble - ARTE
 *
 *
 * @author Aldo
 * @version $Revision: 1 $
 */

public class ARTE extends AbstractClassifier implements MultiClassClassifier,
CapabilitiesHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	@Override
    public String getPurposeString() {
		return "Adaptive random tree ensemble for evolving data stream classification from Paim et al.";
    }

    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
            "ARTEHoeffdingTree", ARTEHoeffdingTree.class,
            "ARTEHoeffdingTree -e 2000000 -g 100 -c 0.01 -n ARTEAttributeClassObserver");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The number of trees.", 100, 1, Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
        "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
        "Total number of concurrent jobs used for processing "
        + "(-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);
    
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
        "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
        "Should use drift detection? If disabled then bkg learner is also disabled");

    public IntOption windowObservationSize  = new IntOption("windowObservationSize", 'w',
            "Size of the observation window to refine statistics and select learners in the voting.", 400, 0, Integer.MAX_VALUE);
        
    
       
    protected static final int SINGLE_THREAD = 0;
    
    protected ARTEBaseLearner[] ensemble;
    protected long instancesSeen;

    transient private ExecutorService executor;
    
    //subspace
    protected Random subspaceRandom;
    protected int  maxValueRandom;
    protected int  minValueRandom;
    
    //statistic window size
    protected double avgAccuracyWindowLearner;
    protected int numAttributes;
    
    
	@Override
	public boolean isRandomizable() {
		return true;
	}
	
	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();
        boolean shouldClassifierVote = true;
        double accWindowLearner = 0.0;
        
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            shouldClassifierVote = (this.ensemble[i].accuracyWindowLearner >= avgAccuracyWindowLearner);

            //updates statistics selection classifier
            this.ensemble[i].updateMatrixConfusion(instance, vote.getArrayRef());
            accWindowLearner += this.ensemble[i].accuracyWindowLearner;
                
            if (vote.sumOfValues() > 0.0  && shouldClassifierVote) { 
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        
        if (accWindowLearner > 0.0) {
        	avgAccuracyWindowLearner = (accWindowLearner / this.ensemble.length);
        }
        return combinedVote.getArrayRef();
	}
	
	
	@Override
	public void resetLearningImpl() {
		// Reset attributes
        this.ensemble = null;
        this.instancesSeen = 0;
        this.avgAccuracyWindowLearner = 0;
        
        // Multi-threading (code inspired by AdaptiveRandomForest)
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1) 
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else 
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent. 
        // this.executor will be null and not used...
        if(numberOfJobs != ARTE.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
		
	}

	
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
        if(this.ensemble == null) 
            initEnsemble(instance);
        
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();   
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            long seed = Double.toString(instancesSeen).hashCode()+i;
            this.ensemble[i].setSeedRandom(seed);
            
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
             if (k > 0) {
            	if(this.executor != null) {
                    TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
                        instance, k, this.instancesSeen);
                    trainers.add(trainer);
                }
                else { // SINGLE_THREAD 
                    this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
                }
            }
        }
        
          
        if(this.executor != null) {
            try {
            	this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
        
	}
	
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		
	}
	
	protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARTEBaseLearner[ensembleSize];
        
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
        
		this.subspaceRandom = new Random();
		this.subspaceRandom.setSeed(n+instance.numClasses());
		this.minValueRandom = 2; 
		this.maxValueRandom = n;
		
		ARTEHoeffdingTree treeLearner = (ARTEHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {
        	
        	treeLearner.subspaceSizeOption.setValue(randomSubSpaceSize());
        	
            this.ensemble[i] = new ARTEBaseLearner(
                i, 
                (ARTEHoeffdingTree) treeLearner.copy(), 
                this.instancesSeen, 
                ! this.disableDriftDetectionOption.isSet(), 
                driftDetectionMethodOption,
                this.windowObservationSize.getValue());
        }
    }
	
	
	private int randomSubSpaceSize() {
		
		int randomSubSpaceSize = this.subspaceRandom.nextInt(maxValueRandom + 1 - minValueRandom) + minValueRandom;
		return randomSubSpaceSize;
	}

	@Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == ARTE.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    @Override
    public Classifier[] getSublearners() {
        /* Extracts the reference to the ARTEHoeffdingTree object from within the ensemble of ARTEBaseLearner's */
        Classifier[] forest = new Classifier[this.ensemble.length];
        for(int i = 0 ; i < forest.length ; ++i)
            forest[i] = this.ensemble[i].classifier;
        return forest;

    }
	
	
	/**
     * Inner class that represents a single tree member of the forest. 
     * It contains some analysis information, such as the numberOfDriftsDetected, 
     */
    protected final class ARTEBaseLearner extends AbstractMOAObject {
        /**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public ARTEHoeffdingTree classifier;
        
        // The drift object parameters. 
        protected ClassOption driftOption;
        
        // Drift detection
        protected ChangeDetector driftDetectionMethod;
        
        public boolean useDriftDetector;
        
        // Statistics
        protected int numberOfDriftsDetected;
        public long instancesLog;
        
        //
        protected int windowObservationSize;
        protected double accuracyWindowLearner; 
        protected Random subspaceRandomBaseLearner;
        
        protected int[] accClassifierArray; 
        protected int lastIndex;
        
        private void init(int indexOriginal, 
        		ARTEHoeffdingTree instantiatedClassifier, 
	            long instancesSeen, 
	            boolean useDriftDetector, 
	            ClassOption driftOption, 
	            int windowObservationSize) {
        	
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            
            this.classifier = instantiatedClassifier;
            this.useDriftDetector = useDriftDetector;
            
            this.numberOfDriftsDetected = 0;
            this.windowObservationSize = windowObservationSize;
            this.instancesLog = 0;

            if(this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }

        }

        public ARTEBaseLearner(int indexOriginal, 
        		ARTEHoeffdingTree instantiatedClassifier, 
        		    long instancesSeen, 
                    boolean useDriftDetector, 
                    ClassOption driftOption, 
                    int windowObservationSize) {
            init(indexOriginal, 
            		instantiatedClassifier, 
            		instancesSeen, 
            		useDriftDetector, 
            		driftOption, 
            		windowObservationSize);
        }

        public void reset() {
        	this.classifier.resetLearning();
        	this.createdOn = instancesSeen;
        	this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            accClassifierArray = null;
            this.classifier.subspaceSizeOption.setValue(randomSubSpaceSizeLocal());
        }

        public void setSeedRandom(long seed) {
        	if (this.subspaceRandomBaseLearner == null)
        		this.subspaceRandomBaseLearner = new Random();
    		this.subspaceRandomBaseLearner.setSeed(seed);
        }
        
        private int randomSubSpaceSizeLocal() {
    		int randomSubSpaceSize = this.subspaceRandomBaseLearner.nextInt(maxValueRandom + 1 - minValueRandom) + minValueRandom;
    		return randomSubSpaceSize;
    	}
        
        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            this.classifier.trainOnInstance(weightedInstance);
            
            // Should it use a drift detector?  
            if(this.useDriftDetector ) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                	this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
                    this.reset();                    
                }
            }
        }

        private void updateMatrixConfusion(Instance instance, double[] vote) {
        	int trueClass = (int) instance.classValue();
            int predictedClass = Utils.maxIndex(vote);
            double acc = 0.0;
            
            if (accClassifierArray == null) {
            	accClassifierArray = new int[this.windowObservationSize+2];
            	lastIndex = -1;
            }
            
            if (lastIndex == this.windowObservationSize-1) {
            	lastIndex = -1;
            	
            }
            	
            if (accClassifierArray[accClassifierArray.length-2] < this.windowObservationSize) {
            	accClassifierArray[accClassifierArray.length-2] ++;
            }
            
            lastIndex++;
            accClassifierArray[accClassifierArray.length-1] -= 
            		accClassifierArray[lastIndex];
            accClassifierArray[lastIndex] = trueClass == predictedClass ? 1 : 0;
            
            
            accClassifierArray[accClassifierArray.length-1] += 
            		accClassifierArray[lastIndex];
            
            acc = (double) accClassifierArray[accClassifierArray.length-1]/accClassifierArray[accClassifierArray.length-2];
            accuracyWindowLearner = acc;
			
		}

		public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
        }
        

    }

    /***
     * Inner class to assist with the multi-thread execution. 
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private ARTEBaseLearner learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;

        public TrainingRunnable(ARTEBaseLearner learner, Instance instance, 
                double weight, long instancesSeen) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
        }

        @Override
        public void run() {
            learner.trainOnInstance(this.instance, this.weight, this.instancesSeen);
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }
    }
    
}