#include "neuralcluster.h"


NeuralCluster::NeuralCluster(int inputs, int outputs, int hidden){

    vector<vector<float> > weightsCreation;
    for(int i = 0; i < inputs+outputs+hidden; i++){
        lastNetActivation.push_back(0.0);
        realNetActivation.push_back(1.0);
        firingStateReal.push_back(0.0);
        firingStateCounter.push_back(0.0);
        differenceVector.push_back(0.0);
        counterActivation.push_back(1.0);
        error.push_back(0.0);
        deltaNet.push_back(0.0);
        lastCounter.push_back(0.0);
        counter_frequence.push_back(0.0);
        counter_pulseActivation.push_back(0.0);
        real_frequence.push_back(0.0);
        real_pulseActivation.push_back(0.0);
        counter_counter.push_back(0);
        real_counter.push_back(0);
        differenceError.push_back(0);
        valueHardeningReal.push_back(0.0);
        valueHardeningCounter.push_back(0.0);
        rawOutputReal.push_back(0.0);
        rawOutputCounter.push_back(0.0);
        vector<float> weightColumn;
        vector<float> momentumColumn;
        for(int j = 0; j < inputs+outputs+hidden; j++){
            weightColumn.push_back(1.0-(2.0*rand()/RAND_MAX));
            momentumColumn.push_back(0.0);
        }
        weights.push_back(weightColumn);
        momentum.push_back(momentumColumn);
    }

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

    for(int i = 0; i < weights.size(); i++){
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
            if((i >= 0)&&(j < numInputs) && (i < numInputs) && (j < numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}
            if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <=numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}
            if((i >= numInputs)&& (j >= numInputs) && (i <= numInputs+numOutputs)&& (j <=numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}
            if(i == j){ weights[i][j] = 0.0; skip = true;}
            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i <= weights.size())&& (j <=weights.size()-1)){ weights[i][j] = 0.0; skip = true;}


            if(!skip){
            //weights[i][j] += ((realNetActivation[j]))*(differenceError[i])*learningRate;

            weights[i][j] += counterActivation[j]*(realNetActivation[i]-counterActivation[i])*counterActivation[i]*(1.0-counterActivation[i])*learningRate;


            //weights[i][j] = weights[i][j]*(1.0-0.1*((counterActivation[j]*(realNetActivation[i]-counterActivation[i]))));

            //weights[i][j] -= ((1.0-counterActivation[j]))*(realNetActivation[i]-counterActivation[i])*learningRate;
            //weights[i][j] -= ((1.0-counterActivation[j]))*(counterActivation[i])*(1.0-counterActivation[i])*(realNetActivation[i]-counterActivation[i])*learningRate;

            //weights[i][j] *= 1.0+counterNet[j]*counterActivation[j];
            //weights[i][j] *= 0.5;
            //weights[i][j] *= (counterActivation[j]*counterActivation[i]);
            //weights[j][i] += (counterActivation[i])*(differenceError[j])*learningRate;
            //weights[j][i] += (counterActivation[j])*(differenceError[i])*learningRate;
            }

        }
        //weights[i][i] += lastCounter[i]*(realNetActivation[i]-counterActivation[i])*learningRate;
    }


}

void NeuralCluster::trainBP(vector<float> target,float learningRate,int iterations){

    for(int i = 0; i < target.size(); i++) error[i] = (target[i]-counterActivation[i]);

    vector<float> newError = error;

    for(int i = 0; i < 16; i++){

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
            weights[i][j] += counterActivation[j]*(error[i])*learningRate;

            if((i >= 0)&& (j >= 0) && (i <= numInputs)&& (j <= numInputs)) weights[i][j] = 0.0;
            if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <= numInputs+numOutputs)) weights[i][j] = 0.0;

            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i <= weights.size()-1)&& (j <= weights.size()-2)) weights[i][j] = 0.0;


            if(i == j) weights[i][j] = 0.0;

        }
    }
}

vector<float> NeuralCluster::getActivation(){
    return counterActivation;
}

vector<float> NeuralCluster::getTarget(){
    return realNetActivation;
}

void NeuralCluster::propergate(vector<float> input,vector<float> output, bool sleep){

    lastCounter = counterActivation;
    lastNetActivation = realNetActivation;

    for(int i = 0; i < input.size(); i++){ if(!sleep){ counterActivation[i] = input[i];} realNetActivation[i] = input[i]; }
    for(int i = input.size(); i < output.size()+input.size(); i++) {  realNetActivation[i] = output[i-input.size()];  }

    vector<float> deltaEnergysI;
    vector<float> deltaEnergys;

        for(int i = 0; i < weights.size(); i++){
            float x = 1.0;
            float y = 1.0;
            float weightsSum = 1.0;
            for(int j = 0; j < weights[i].size(); j++){
                if(weights[i][j] != 0.0){
                     float deltaEnergyReal =  (weights[i][j]*(realNetActivation[j]));
                     float deltaEnergyCounter =  (weights[i][j]*(counterActivation[j]));
                     x += deltaEnergyReal;
                     y += deltaEnergyCounter;
                     weightsSum *= weights[i][j];
                }
            }
            deltaEnergysI.push_back(1.0/(1.0+exp(-x)));
            deltaEnergys.push_back((1.0/(1.0+exp(-y))));

            //deltaEnergysI.push_back(x/weightsSum);
            //deltaEnergys.push_back(y/weightsSum);

            //counterNetActivationDerivative[i] = cos(x);
        }


        for(int i = input.size(); i < weights.size()-1; i++){
            //counterActivation[i] += deltaEnergys[i]*(1.0-counterActivation[i])*counterActivation[i];
            /*
            if(deltaEnergys[i]-0.5 > 0.0) counterActivation[i] += (deltaEnergys[i]-0.5)*(1.0-counterActivation[i]);
            else counterActivation[i] += (deltaEnergys[i]-0.5)*(counterActivation[i]);
            */
            valueHardeningCounter[i] = (valueHardeningCounter[i]+abs(counterActivation[i]-realNetActivation[i]))/2.0;
            counterActivation[i] = (valueHardeningCounter[i])*counterActivation[i]+(1.0-valueHardeningCounter[i])*deltaEnergys[i];
            //counterActivation[i] = deltaEnergys[i];

            //counterActivation[i] = deltaEnergys[i];

        }

        for(int i = 0; i < weights.size(); i++){

            /*
            counter_pulseActivation[i] = 0.0;
            if(counterActivation[i] > 1.0*rand()/RAND_MAX) counter_pulseActivation[i]=1.0;


            if((counter_counter[i]>=(counter_frequence[i]+1))){
                counter_frequence[i] = (int)(8.0-counterActivation[i]*8.0);
                counter_pulseActivation[i] = 1.0;
                counter_counter[i] = 0;
            }else{
                counter_pulseActivation[i] = 0.0;
            }
            counter_counter[i]++;
            */
        }

        for(int i = input.size()+output.size(); i < weights.size()-1; i++){

            /*
            if(deltaEnergysI[i]-0.5 > 0.0) realNetActivation[i] += (deltaEnergysI[i]-0.5)*(1.0-realNetActivation[i]);
            else realNetActivation[i] += (deltaEnergysI[i]-0.5)*(realNetActivation[i]);
            */
            valueHardeningReal[i] = (valueHardeningReal[i]+abs(realNetActivation[i]-counterActivation[i]))/2.0;
            realNetActivation[i] = (valueHardeningReal[i])*realNetActivation[i]+(1.0-valueHardeningReal[i])*deltaEnergysI[i];
            //realNetActivation[i] = deltaEnergysI[i];
            //realNetActivation[i] = deltaEnergysI[i];

            //realNetActivation[i] = deltaEnergysI[i];
            //realNetActivation[i] += deltaEnergysI[i]*(1.0-realNetActivation[i])*realNetActivation[i];


        }

        for(int i = 0; i < weights.size(); i++){

            /*
            real_pulseActivation[i] = 0.0;
            if(realNetActivation[i] > 1.0*rand()/RAND_MAX) real_pulseActivation[i]=1.0;


            if((real_counter[i]>=(real_frequence[i]+1))){
                real_frequence[i] = (int)(8.0-realNetActivation[i]*8.0);
                real_pulseActivation[i] = 1.0;
                real_counter[i] = 0;
                //cout << "fire" <<real_frequence[i];
            }else{
                real_pulseActivation[i] = 0.0;
            }
            real_counter[i]++;
            */

        }

        for(int i = 0; i < input.size(); i++){ if(!sleep){ counterActivation[i] = input[i];} realNetActivation[i] = input[i]; }
        for(int i = input.size(); i < output.size()+input.size(); i++) {  realNetActivation[i] = output[i-input.size()];  }




}
