#include "neuralcluster.h"


NeuralCluster::NeuralCluster(int inputs, int outputs, int hidden){

    vector<vector<float> > weightsCreation;
    for(int i = 0; i < inputs+outputs+hidden; i++){
        fireCounter.push_back(1.0);
        counterActivation.push_back(1.0);
        lastCounter.push_back(1.0);
        polarityCounter.push_back(1.0);
        fireReal.push_back(1.0);
        realActivation.push_back(1.0);
        lastReal.push_back(1.0);
        polarityReal.push_back(1.0);
        period.push_back(rand()%maxPeriod+1);
        counter.push_back(rand()%period[i]+1);
        slowness.push_back(1.0);
        derived.push_back(0.0);
        if(i > inputs+outputs) slowness[i] = 1.0*rand()/RAND_MAX;
        vector<float> weightColumn;
        for(int j = 0; j < inputs+outputs+hidden; j++){
            //if((j+i)%2 ==0) weightColumn.push_back(-0.001);
            //else  weightColumn.push_back(0.001);

            //weightColumn.push_back(0.01);

            weightColumn.push_back(0.1*((1.0*rand()/RAND_MAX-0.5)));

        }
        weights.push_back(weightColumn);
        error.push_back(0.0);
    }

    fireCounter[fireCounter.size()-1] = 1.0;
    fireReal[fireReal.size()-1] = 1.0;


    numInputs = inputs;
    numOutputs = outputs;
    numHiddens = hidden;

}

vector<vector<float>> NeuralCluster::getWeights(){
    return weights;
}

void NeuralCluster::train(float learningRate){
/*
    float maxVal = 0.0;
    vector<int> neuronsChain;
    int maxChainLength = 16;

    int length = rand()%maxChainLength;

    neuronsChain.push_back(rand()%weights.size());
    for(int i = 0; i < length; i++){
        int k = rand()%weights.size();
        while(weights[neuronsChain[i]][k] == 0.0){
            k = rand()%weights.size();
        }
        neuronsChain.push_back(k);
    }

    float ErrorSum = 0.0;
    for(int i = 0; i < neuronsChain.size()-1; i++){
        ErrorSum += (realNetActivation[neuronsChain[i]]-counterActivation[neuronsChain[i]]);
    }

    for(int i = 1; i < neuronsChain.size(); i++){
        weights[neuronsChain[i-1]][neuronsChain[i]] += counterActivation[neuronsChain[i]] * ErrorSum  * counterActivation[neuronsChain[i-1]] * (1.0-counterActivation[neuronsChain[i-1]]);
    }
*/

    float absMax = 0.0;
    for(int i = 0; i < weights.size()-1; i++){
        for(int j = 0; j < weights[i].size(); j++){
            //momentum[i][j] += lastCounter[j]*(realNetActivation[i]-counterActivation[i])*(counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //momentum[i][j] -= (1.0-lastCounter[j])*(realNetActivation[i]-counterActivation[i])*(counterActivation[i])*(1.0-counterActivation[i])*learningRate;


            //weights[j][i] += ((lastCounter[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //weights[i][j] -= ((2.0-lastCounter[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
          //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += (((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i]))*learningRate;
            //weights[i][j] -= (1.0-(lastCounter[j]))*(realNetActivation[i]-counterActivation[i])*counterActivation[i]*(1.0-counterActivation[i])*learningRate;

            //deltaSynapse[i][j] = realNetActivation[j]*weights[i][j]*realNetActivation[i]-counterActivation[j]*weights[i][j]*counterActivation[i];

            bool skip = false;

            //if((i >= 0)&&(j < numInputs) && (i < numInputs) && (j < numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}
            //if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <=numInputs)){ weights[i][j] = 0.0; skip = true;}
            //if((i >= numInputs)&& (j >= numInputs) && (i <= numInputs+numOutputs)&& (j <=numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}
            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}

            //if(i == j){ weights[i][j] = 0.0; skip = true;}


            if(!skip){

                //weights[i][j] *= counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]))*learningRate*0.001+1.0;
                //weights[i][j] *= 1.0+counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]));
                weights[i][j] += (lastReal[j])*(realActivation[i]-lastReal[i])*lastReal[i]*(1.0-lastReal[i]);
                //weights[j][i] += (lastReal[j])*(realActivation[i]-lastReal[i])*lastReal[i]*(1.0-lastReal[i])*2.5;
            }

        }
        //weights[i][i] += lastCounter[i]*(realNetActivation[i]-counterActivation[i])*learningRate;
    }


}

void NeuralCluster::trainBP(vector<float> target,float learningRate,int iterations){

    for(int i = 0; i < target.size(); i++) error[i] = (target[i]-counterActivation[i])*(counterActivation[i])*(1.0-counterActivation[i]);

    vector<float> newError = error;

    for(int i = 0; i < 4; i++){

    for(int i = numInputs+numOutputs; i < weights.size(); i++){
        float bpError = 0.0;
        for(int j = 0; j < weights[i].size(); j++){
                 bpError +=  weights[i][j]*(error[j]);
        }
        newError[i] = (bpError*(counterActivation[i])*(1.0-counterActivation[i]));
    }

    }

    error = newError;
    float maxVal = 0.0;
    for(int i = 0; i < weights.size(); i++){
        for(int j = 0; j < weights[i].size(); j++){
            weights[i][j] += fireCounter[j]*(error[i])*learningRate;

            //if((i >= 0)&& (j >= 0) && (i <= numInputs)&& (j <= numInputs)) weights[i][j] = 0.0;
            //if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <= numInputs+numOutputs)) weights[i][j] = 0.0;
            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i <= weights.size()-1)&& (j <= weights.size()-2)) weights[i][j] = 0.0;


            //if(i == j) weights[i][j] = 0.0;

        }
    }
}

float NeuralCluster::signum(float x){
    if(x > 0.0) return 1.0;
    else return -1.0;
}

vector<float> NeuralCluster::getActivation(){
    return counterActivation;
}

vector<float> NeuralCluster::getTarget(){
    return realActivation;
}

void NeuralCluster::propergate(vector<float> input,vector<float> output, bool sleep){
    lastCounter = counterActivation;
    lastReal = realActivation;

    for(int i = 0; i < input.size(); i++){ if(!sleep){ counterActivation[i] = input[i]; fireCounter[i] = input[i]; } realActivation[i] = input[i]; fireReal[i] = input[i];}
    for(int i = input.size(); i < output.size()+input.size(); i++) {  realActivation[i] = output[i-input.size()]; fireReal[i] =output[i-input.size()];  }

    vector<float> deltaEnergysI;
    vector<float> deltaEnergys;

        for(int i = 0; i < weights.size()-1; i++){
            float x = 1.0;
            float y = 1.0;
            float weightsSum = 1.0;
            for(int j = 0; j < weights[i].size(); j++){
                     float deltaEnergyReal =  (weights[i][j]*(fireReal[j]));
                     float deltaEnergyCounter =  (weights[i][j]*(fireCounter[j]));
                     x += deltaEnergyReal;
                     y += deltaEnergyCounter;
                     weightsSum *= weights[i][j];
            }

            deltaEnergysI.push_back(1.0/(1.0+exp(-x)));
            deltaEnergys.push_back(1.0/(1.0+exp(-y)));

            //deltaEnergysI.push_back(x/weightsSum);
            //deltaEnergys.push_back(y/weightsSum);

            //counterNetActivationDerivative[i] = cos(x);
            //slowness[i] = 1.0*rand()/RAND_MAX;
        }


        for(int i = input.size(); i < weights.size()-1; i++){

            /*
            if(i < numInputs+numOutputs){
            if(counter[i]%period[i] == 0)slowness[i] = 1.0;
            else slowness[i] = 0.0;
            }
            */


            counterActivation[i] = (counterActivation[i]*(1.0-slowness[i])+deltaEnergys[i]*slowness[i]);
            /*
            if(counterActivation[i] > rand()*1.0/RAND_MAX)fireCounter[i] = 1.0;
            else fireCounter[i] = 0.0;
            */

            fireCounter[i] = counterActivation[i];
            //fireCounter[i] = counterActivation[i];// = ((deltaEnergys[i]+minVal)/(maxVal+minVal));

            //fireCounter[i] = counterActivation[i] = deltaEnergys[i]/(sumActivation/weights.size());

        }

        for(int i = input.size()+output.size(); i < weights.size()-1; i++){

            /*
            if(counter[i]%period[i] == 0)slowness[i] = 1.0;
            else slowness[i] = 0.0;
            */

            realActivation[i] = (realActivation[i]*(1.0-slowness[i])+deltaEnergysI[i]*slowness[i]);
            //float rand = rand()*1.0/RAND_MAX;
            //fireReal[i] = (1.0-realActivation[i])*rand+realActivation[i]*(1.0-rand);
            /*
            if(realActivation[i] > rand()*1.0/RAND_MAX) fireReal[i] = 1.0;
            else fireReal[i] = 0.0;
            */

            fireReal[i] = realActivation[i];// = ((deltaEnergysI[i]+minValI)/(maxValI+minValI));

            //fireReal[i] = realActivation[i] = deltaEnergysI[i]/(sumActivationI/weights.size());

            //counter[i]++;

        }


}
