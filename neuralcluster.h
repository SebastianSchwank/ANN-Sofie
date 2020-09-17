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
    NeuralCluster(int inputs, int outputs, int hidden, int rekurrent);

    void propergate(vector<float> input, vector<float> output, bool sleep, bool hiddenWrite);
    vector<vector<float>> getWeights();
    void train(float learningRate);
    void trainBP(vector<float> target,float learningRate,int iterations);
    vector<float> getTarget();
    float signum(float x);
    float minMax(float x);

    vector<float> getActivation();

    void syncronize();


private:

    int                   maxPeriod = 1;

    vector<float>         fireCounter;
    vector<float>         counterActivation;
    vector<float>         lastCounter;
    vector<float>         polarityCounter;
    vector<float>         fireReal;
    vector<float>         realActivation;
    vector<float>         lastReal;
    vector<float>         polarityReal;
    vector<float>         derived;

    vector<int>           counter;
    vector<int>           period;

    vector<float>         slowness;

    vector<float>         slope;
    vector<vector<float>> weights;
    vector<vector<float>> momentum;
    int                   numInputs,numOutputs,numHiddens,numRekurrent;

    vector<float>         error;

    float maxResultReal = 1.0;
    float maxResultCounter = 1.0;

};

#endif // NEURALCLUSTER_H
