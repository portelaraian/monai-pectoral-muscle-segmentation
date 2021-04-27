gpu=0

test() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python3 ./src/cnn/main.py test ${conf} --gpu ${gpu}
}

test model002
