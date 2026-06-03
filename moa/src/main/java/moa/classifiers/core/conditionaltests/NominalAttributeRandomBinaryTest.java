package moa.classifiers.core.conditionaltests; 


import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.StringUtils; 

public class NominalAttributeRandomBinaryTest extends InstanceConditionalTest {

    private static final long serialVersionUID = 1L;
    
    protected int attIndex;
    
    protected boolean[] subSetAMask;

    public NominalAttributeRandomBinaryTest(int attIndex, boolean[] subSetAMask) {
        this.attIndex = attIndex; 
        this.subSetAMask = subSetAMask;
    }

    
    @Override
    public int branchForInstance(Instance inst) {
        if (inst.isMissing(this.attIndex)) {
            return -1; 
        }
        int attValue = (int) inst.value(this.attIndex);
        
        if (attValue >= 0 && attValue < this.subSetAMask.length && this.subSetAMask[attValue]) {
            return 0; // Branch 0
        }
        return 1; // Branch 1
    }

    @Override
    public int maxBranches() {
        return 2; // Split is always binary.
    }

    @Override
    public String describeConditionForBranch(int branch, InstancesHeader context) {
        if (branch == 0) {
            return context.attribute(this.attIndex).name() + " is in SubSet A (Random-Binary)";
        }
        return context.attribute(this.attIndex).name() + " is in SubSet B (Random-Binary)";
    }

    @Override
    public int[] getAttsTestDependsOn() {
        return new int[]{this.attIndex};
    }
    

    public boolean isNominalTest() {
        return true;
    }

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		StringUtils.appendIndented(sb, indent, "Nominal random binary test");
		
	}
    
}