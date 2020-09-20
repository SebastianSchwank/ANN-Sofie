#include "neuralcluster.h"


NeuralCluster::NeuralCluster(int inputs, int outputs, int hidden, int attention){

    vector<vector<float> > weightsCreation;
    for(int i = 0; i < inputs+outputs+hidden+attention+1; i++){
        fireCounter.push_back(1.0);
        counterActivation.push_back(1.0);
        lastCounter.push_back(1.0);
        polarityCounter.push_back(1.0);
        fireReal.push_back(1.0);
        realActivation.push_back(1.0);
        lastReal.push_back(1.0);
        beforelastReal.push_back(1.0);
        polarityReal.push_back(1.0);
        period.push_back(rand()%maxPeriod+1);
        counter.push_back(rand()%period[i]+1);
        slowness.push_back(1.0);
        derived.push_back(0.0);
        //if(i > inputs+outputs) slowness[i] = 1.0*rand()/RAND_MAX;
        //slowness[i] = slowness[i]*slowness[i]*slowness[i];
        vector<float> weightColumn;
        vector<float> momentumColumn;
        for(int j = 0; j < inputs+outputs+hidden+attention+1; j++){
            //if((j+i)%2 ==0) weightColumn.push_back(-0.001);
            //else  weightColumn.push_back(0.001);

            //weightColumn.push_back(0.01);

            weightColumn.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
            momentumColumn.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));

        }
        weights.push_back(weightColumn);
        momentum.push_back(momentumColumn);
        slope.push_back(1.0);
        error.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
        lastError.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
        beforeLasteError.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
        derivedError.push_back(0.0);
    }

    fireCounter[fireCounter.size()-1] = 1.0;
    fireReal[fireReal.size()-1] = 1.0;


    numInputs = inputs;
    numOutputs = outputs;
    numHiddens = hidden;
    numRekurrent = attention;

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

    beforeLasteError = lastError;
    lastError = error;
    float absMax = 0.0;
    for(int i = 0; i < weights.size(); i++){
        //if((i >= numInputs) && (i < numInputs+numOutputs+numHiddens)) error[i] = (fireReal[i]-fireCounter[i]);
        //if(i < numInputs) error[numInputs+numOutputs+numHiddens+i] = (fireReal[i]-fireCounter[i]);
        error[i] = (fireReal[i]-fireCounter[i]);
        //if(i > numInputs+numOutputs+numHiddens) error[i] = (fireReal[i+numInputs+numOutputs+numHiddens]-fireCounter[i+numInputs+numOutputs+numHiddens]);

        derivedError[i] = (error[i])/((error[i]*lastError[i]));
        if(derivedError[i] != derivedError[i]) derivedError[i] = 0.0;
        //cout << i << ":" << derivedError[i] << " ";
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
            //if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <=numInputs)){ weights[i][j] = 0.0; skip = true;}
            if((i >= numInputs)&& (j >= numInputs) && (i <= numInputs+numOutputs)&& (j < numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (j <= numInputs+numOutputs+numHiddens) && (i <= numInputs+numOutputs+numHiddens)){ weights[i][j] = 0.0; skip = true;}

            if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (j <= numInputs+numOutputs+numHiddens) && (i <= numInputs+numOutputs+numHiddens)){ weights[i][j] = 0.0; skip = true;}

            if((i >= numInputs+numOutputs+numHiddens)&& (j >= numInputs+numOutputs+numHiddens) && (j <= numInputs+numOutputs+numHiddens+numRekurrent-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent)){ weights[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs+numHiddens+numRekurrent)&& (j >= numInputs+numOutputs+numHiddens+numRekurrent) && (j <= numInputs+numOutputs+numHiddens+numRekurrent*2-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent*2)){ weights[i][j] = 0.0; skip = true;}

            //if((i > numInputs+numOutputs+numHiddens/2)&& (j > numInputs+numOutputs+numHiddens/2) && (j <= numInputs+numOutputs+numHiddens-1) && (i <= numInputs+numOutputs+numHiddens)){ weights[i][j] = 0.0; skip = true;}
            //if((i > numInputs+numOutputs+numHiddens)&& (j > numInputs+numOutputs+numHiddens) && (j <= numInputs+numOutputs+numHiddens+numRekurrent-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent)){ weights[i][j] = 0.0; skip = true;}


            //if(i == j){ weights[i][j] = 0.0; skip = true;}

            //if(rand()%16 != 0) skip = true;


            if(!skip){

                //weights[i][j] *= counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]))*learningRate*0.001+1.0;
                //weights[i][j] *= 1.0+counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]));

                float currentError = ((1.0-(fireCounter[j])*(fireCounter[j])*error[i]*error[i]*0.25));
                derived[i] = ((error[i])/((2.0-error[i])*(2.0+error[i])));
                if(derived[i] != derived[i]) derived[i] = 0.0;
                float derivedError = ((error[i])/((2.0-error[i])*(2.0+error[i])));

                momentum[i][j] *= currentError;
                momentum[i][j] *= 0.99;
                momentum[i][j] += (fireCounter[j])*(error[i])*learningRate;
                weights[i][j] +=  (fireCounter[j])*(error[i])*learningRate+momentum[i][j]*0.3;



                //weights[j][i] += (lastReal[j])*(realActivation[i]-lastReal[i])*0.01;
            }

        }
        //weights[i][i] += lastCounter[i]*(realNetActivation[i]-counterActivation[i])*learningRate;
    }


}

void NeuralCluster::trainBP(vector<float> target,float learningRate,int iterations){

    for(int i = 0; i < target.size(); i++) error[i] = (target[i]-fireCounter[i]);

    vector<float> newError = error;

    for(int i = 0; i < iterations; i++){

    for(int i = numInputs+numOutputs; i < weights.size()-1; i++){
        float bpError = 0.0;
        for(int j = 0; j < weights[i].size(); j++){
                 bpError +=  weights[i][j]*(newError[j]);
        }
        newError[i] = (bpError);
    }

    }

    float maxVal = 0.0;
    for(int i = 0; i < weights.size()-1; i++){
        for(int j = 0; j < weights[i].size(); j++){
            weights[i][j] += fireCounter[j]*(newError[i])*learningRate;

            if((i >= 0)&& (j >= 0) && (i <= numInputs)&& (j <= numInputs)) weights[i][j] = 0.0;
            if((i >= numInputs)&& (j >= 0) && (i <= numInputs+numOutputs)&& (j <= numInputs+numOutputs)) weights[i][j] = 0.0;
            if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (j <= weights.size()-2)) weights[i][j] = 0.0;


            if(i == j) weights[i][j] = 0.0;

        }
    }
}

float NeuralCluster::signum(float x){
    if(x > 0.0) return 1.0;
    else return -1.0;
}

vector<float> NeuralCluster::getActivation(){
    return fireCounter;
}

vector<float> NeuralCluster::getTarget(){
    return fireReal;
}

float NeuralCluster::minMax(float x){

    if(x <= 0.0) x = -1.0+exp(x);
    if(x >= 0.0) x = 1.0-exp(-x);
    //x = 1.0/(1.0+exp(-x));

    return x;
}

void NeuralCluster::propergate(vector<float> input,vector<float> output, bool sleep, bool hiddenWrite){
    lastCounter = fireCounter;
    lastReal = fireReal;

    for(int i = 0; i < input.size(); i++){  counterActivation[i] = input[i]; fireCounter[i] = input[i]; realActivation[i] = input[i]; fireReal[i] = input[i];}
    for(int i = input.size(); i < output.size()+input.size(); i++) {  realActivation[i] = output[i-input.size()]; fireReal[i] =output[i-input.size()];  }


    vector<float> deltaEnergysI;
    vector<float> deltaEnergys;
    vector<float> deltaError;

    float sumCounterClassyfier = 0;

        for(int i = 0; i < weights.size()-1; i++){
            float x = 0.0;
            float y = 0.0;
            float errorSum = 0.0;
            for(int j = 0; j < weights[i].size(); j++){


                     float deltaEnergyReal =  weights[i][j]*(fireReal[j]);
                     float deltaEnergyCounter =  weights[i][j]*(fireCounter[j]);

                     x += deltaEnergyReal;
                     y += deltaEnergyCounter;

                     //sumWeights += minMax(weights[i][j]);
            }
            /*
            derived[i] = cos(y);

            x = sin(x);
            y = sin(y);
            */



            //deltaError.push_back(deltaErr);

            //derived[i] = cos(y);

            if(maxResultReal < (minMax(x))) maxResultReal = (minMax(x));
            if(maxResultCounter < (minMax(y))) maxResultCounter = (minMax(y));

            deltaEnergysI.push_back(minMax(x));
            deltaEnergys.push_back(minMax(y));

            //derived[i] = exp(-y);
            //if(deltaEnergys[i] == 0) derived[i] = 0.0;


            //deltaEnergysI.push_back(x/weightsSum);
            //deltaEnergys.push_back(y/weightsSum);

            //counterNetActivationDerivative[i] = cos(x);
            //slowness[i] = 1.0*rand()/RAND_MAX;
        }

        for(int i = numInputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++) fireCounter[i] = (deltaEnergys[i])/1.0;

        for(int i = numInputs+numOutputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++) fireReal[i] = (deltaEnergysI[i])/1.0;

        if(hiddenWrite) for(int i = 0; i < numInputs; i++) fireCounter[numInputs+numOutputs+numHiddens+i] = deltaEnergys[numInputs+numOutputs+numHiddens+i]*input[i];
        if(hiddenWrite) for(int i = 0; i < numInputs; i++) fireReal[numInputs+numOutputs+numHiddens+i] = deltaEnergysI[numInputs+numOutputs+numHiddens+i]*input[i];


}
