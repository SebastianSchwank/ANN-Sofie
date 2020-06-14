#ifndef NEURALCLUSTER_H
#define NEURALCLUSTER_H


#include <vector>
#include <iostream>
#include <cmath>
#include <QDebug>

//Ideas:
//Probailistic Firering
//Dropout
//Momentum

using namespace std;

class NeuralCluster
{
public:
    NeuralCluster(int inputs, int outputs, int hidden);

    void propergate(vector<float> input, vector<float> output, bool sleep);
    vector<vector<float>> getWeights();
    void train(float learningRate);
    void trainBP(vector<float> target,float learningRate);
    vector<float> getTarget();

    vector<float> getActivation();

    void syncronize();


private:
    vector<int>           last_pulse_activation;
    vector<int>           real_counter;
    vector<int>           counter_counter;
    int                   numInputs;
    int                   numOutputs;
    int                   numHiddens;
    vector<vector<float>> weights;
    vector<vector<float>> momentum;
    vector<vector<float>> deltaSynapse;
    vector<float>         counterActivation;
    vector<float>         lastCounter;
    vector<float>         realNetActivation;
    vector<float>         deltaNet;
    vector<float>         NetActivation;
    vector<float>         lastNetActivation;
    vector<float>         beforelastNetActivation;
    vector<float>         errorNet;
    vector<float>         realNetActivationSum;
    vector<float>         counterNet;
    vector<float>         lastError;
    vector<float>         firingStateReal;
    vector<float>         firingStateCounter;
    vector<float>         differenceVector;
    vector<float>         error;
    vector<float>         deriveReal;
    vector<float>         deriveCounter;
    vector<int>           counter_frequence;
    vector<float>           counter_pulseActivation;
    vector<int>           real_frequence;
    vector<int>           real_pulseActivation;
    vector<float>         differenceError;
    vector<float>         firingRate_real;
    vector<float>         firingRate_counter;
    vector<float>         meanActivationReal;
    vector<float>         meanActivationCounter;

};

#endif // NEURALCLUSTER_H
