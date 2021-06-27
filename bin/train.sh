gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python3 ./src/cnn/main.py train ${conf} --gpu ${gpu}
}

train model015

