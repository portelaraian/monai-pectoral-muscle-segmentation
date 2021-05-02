gpu=0

test() {
    model=$1
    _snapshot=$2

    conf=./conf/${model}.py
    snapshot=./model/${model}/${_snapshot}
    test=./model/${model}/output/${_snapshot}_test.pkl
    
    python3 ./src/cnn/main.py test ${conf} --snapshot ${snapshot} --output ${test} --gpu ${gpu}
}

# test model002 model_key_metric=0.9104.pt
#test model003 SegResNet_focalLoss_fold0_0.9094.pt
#test model003 SegResNet_focalLoss_fold1_0.9342.pt
#test model003 SegResNet_focalLoss_fold2_0.9309.pt
#test model003 SegResNet_focalLoss_fold3_0.9181.pt
#test model003 SegResNet_focalLoss_fold4_0.9368.pt

test model003