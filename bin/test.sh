gpu=0

test() {
    model=$1
    
    conf=./conf/${model}.py
    python3 ./src/cnn/main.py test ${conf} --gpu ${gpu}
}

test model016