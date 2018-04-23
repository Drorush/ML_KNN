package HomeWork3;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import java.util.*;

class DistanceCalculator {

    protected boolean m_Efficient = false;

    /**
    * We leave it up to you whether you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it as a class variables.
    */
    public double distance (Instance one, Instance two, int p, double kNeighborDist) {
        if (!m_Efficient)
        {
            return (p > 3) ? lInfinityDistance(one,two) : lpDisatnce(one, two, p);
        }
        // if we are in the efficient calculations
        return (p > 3) ? efficientLInfinityDistance(one,two, kNeighborDist) : efficientLpDisatnce(one,two, p, kNeighborDist);
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDisatnce(Instance one, Instance two, int p) {
        int d = one.numAttributes()-1; // the dimension of the vector
        double distance = 0;
        double difference;

        for (int i = 0; i < d; i++)
        {
            difference = one.value(i) - two.value(i);
            distance += Math.pow(Math.abs(difference),p);
        }

        switch(p)
        {
            case 1: break;
            case 2: distance = Math.sqrt(distance);
                break;
            case 3: distance = Math.cbrt(distance);
                break;
        }

        return distance;
    }

    /**
     * Returns the Lp infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        int d = one.numAttributes()-1; // the dimension of the vector
        double max = Double.MIN_VALUE;
        double difference;

        for (int i = 0; i < d; i++)
        {
            difference = Math.abs((one.value(i)-two.value(i)));
            max = (difference > max) ? difference : max;
        }

        return max;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDisatnce(Instance one, Instance two, int p, double kNeighborDist) {
        int d = one.numAttributes()-1; // the dimension of the vector
        double distance = 0;
        double difference;

        for (int i = 0; i < d; i++)
        {
            difference = one.value(i) - two.value(i);
            distance += Math.pow(Math.abs(difference),p);
            if (distance > kNeighborDist) break;
        }

        switch(p)
        {
            case 1: break;
            case 2: distance = Math.sqrt(distance);
                break;
            case 3: distance = Math.cbrt(distance);
                break;
        }

        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double kNeighborDist) {
        int d = one.numAttributes()-1; // the dimension of the vector
        double max = Double.MIN_VALUE;
        double difference;

        for (int i = 0; i < d; i++)
        {
            difference = Math.abs((one.value(i)-two.value(i)));
            max = (difference > max) ? difference : max;
            if (max > kNeighborDist) break;
        }

        return max;
    }
}

public class Knn implements Classifier {

    private double m_kNeighborDist = 0;
    protected boolean distEffCheck = false;
    private Instances m_trainingInstances;
    private int k; // {1,2,...,20}
    private int p; // {1,2,3,infinity} // p = 4 means infinity
    private int m_weightingScheme; // 0 for uniform , 1 for weighted

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_trainingInstances = instances;
    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        HashSet<Instance> set = findNearestNeighbors(instance);
        Iterator<Instance> it = set.iterator();
        double size = set.size();
        // weightingScheme = 0 for uniform and 1 for weighted
        return (m_weightingScheme == 0) ? getAverageValue(instance, it, size) : getWeightedAverageValue(instance, it);
    }

    /**
     * Calculates the average error on a given set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     * @param insatnces
     * @return
     */
    public double calcAvgError (Instances insatnces){
        double error = 0;
        double singleError;
        Instance inst;
        for (int i = 0; i < insatnces.size(); i++)
        {
            inst = insatnces.instance(i);
            singleError = inst.value(m_trainingInstances.classIndex()) - regressionPrediction(inst);
            error += Math.abs(singleError);
        }


        return error/((double)insatnces.size());
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param arr Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances[] arr, int num_of_folds, int validationIndex){
        return calcAvgError(arr[validationIndex]);
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public HashSet<Instance> findNearestNeighbors(Instance instance) {
        DistanceCalculator distCalc = new DistanceCalculator();
        distCalc.m_Efficient = distEffCheck;
        // this map will hold all the instances as keys and their distances from instace as value
        Map<Instance, Double> map = new HashMap<>();
        HashSet<Instance> kNeighbors = new HashSet<>();

        return distEffCheck ? efficientNearestNeighbors(instance, map, kNeighbors, distCalc) :unEfficientFindNearestNeighbors(instance, kNeighbors, distCalc, map);
    }

    private HashSet<Instance> efficientNearestNeighbors(Instance instance, Map<Instance, Double> map, HashSet<Instance> kNeighbors, DistanceCalculator distCalc)
    {
        Instance secondInstance;
        double dist;
        double kNeighborDistance = Double.MIN_VALUE;
        Instance kNeighbor = null;
        distCalc.m_Efficient = false;
        // at first get k neighbors
        for (int i = 0; i < k; i++)
        {
            secondInstance = m_trainingInstances.instance(i);
            dist = distCalc.distance(instance, secondInstance, p, kNeighborDistance);
            map.put(secondInstance, dist);
            if (dist > kNeighborDistance) {
                kNeighborDistance = dist;
                kNeighbor = secondInstance;
            }
        }

        distCalc.m_Efficient = true;
            // check efficiently the dist of rest of the instances.
            for (int j = k; j < m_trainingInstances.size(); j++)
            {
                secondInstance = m_trainingInstances.instance(j);
                dist = distCalc.distance(instance,secondInstance,p, kNeighborDistance);
                if (dist < kNeighborDistance)
                {
                    map.remove(kNeighbor);
                    map.put(secondInstance, dist);
                    // find the new kNeighbor and his dist
                    kNeighborDistance = Double.MIN_VALUE;
                    for (Instance key : map.keySet())
                    {
                        dist = map.get(key);
                        if (dist > kNeighborDistance)
                        {
                            kNeighbor = key;
                            kNeighborDistance = dist;
                        }
                    }
                }
            }

            // return the chosen k neighbors
            for (Instance key : map.keySet())
            {
                kNeighbors.add(key);
            }

        m_kNeighborDist = kNeighborDistance; // set the kNeighborDist for other methods
        return kNeighbors;
    }

    private HashSet<Instance> unEfficientFindNearestNeighbors(Instance instance, HashSet<Instance> kNeighbors, DistanceCalculator distCalc, Map<Instance,Double> map)
    {
        Instance secondInstance;
        InstanceComparator ic = new InstanceComparator();

        for (int i = 0; i < m_trainingInstances.size(); i++)
        {

            secondInstance = m_trainingInstances.instance(i);
            if(ic.compare(instance,secondInstance) != 0)
            {

                map.put(secondInstance, distCalc.distance(instance, secondInstance, p, 0));
            }
        }

        // get k neighbors
        for(int i = 0; i < k; i++)
        {
            kNeighbors.add(getAndRemoveMinKey(map, map.keySet()));
        }

        return kNeighbors;
    }


    /** getting the instance with min value, extracting him from map and returns him **/
    private Instance getAndRemoveMinKey(Map<Instance, Double> map, Set<Instance> keys) {
        Instance minKey = null;
        double value;
        double minValue = Double.MAX_VALUE;
        for(Instance key : keys)
        {
            value = map.get(key);
            if(value < minValue)
            {
                minValue = value;
                minKey = key;
            }
        }

        map.remove(minKey);
        return minKey;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (Instance instance, Iterator<Instance> it, double size)
    {
        double value = 0;
        while (it.hasNext())
        {
            value += it.next().value(m_trainingInstances.classIndex());
        }

        return (value/size);
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Instance instance, Iterator<Instance> it) {
        Instance inst;
        DistanceCalculator dcalc = new DistanceCalculator();
        dcalc.m_Efficient = distEffCheck;
        double numerator = 0;
        double denominator = 0;
        double wi;
        double dist = 0;

        while (it.hasNext())
        {
            inst = it.next();
            dist = dcalc.distance(instance, inst, p, m_kNeighborDist);
            if (dist == 0)
            {
                return inst.value(m_trainingInstances.classIndex());
            }

            wi = 1.0 / Math.pow(dist,2);
            if (wi > 0)
            {
                numerator += wi * inst.value(m_trainingInstances.classIndex());
                denominator += wi;
            }
        }


        return (numerator/denominator);
    }

    public void setK(int k)
    {
        this.k = k;
    }

    public void setP(int p)
    {
        this.p = p;
    }

    public void setWeightingScheme(int weightingScheme)
    {
        m_weightingScheme = weightingScheme;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
