gpu=0

test() {
    model=$1
    _snapshot=$2

    conf=./conf/${model}.py
    snapshot=./model/${model}/${_snapshot}
    test=./model/${model}/output/${_snapshot}_test.pkl
    
    python3 ./src/cnn/main.py test ${conf} --snapshot ${snapshot} --output ${test} --gpu ${gpu}
}

#test model002
#test model003
#test model004
#test model006
#test model007
#test model008
#test model009
#test model010
#test model0031