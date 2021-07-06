gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python3 ./src/cnn/main.py train ${conf} --gpu ${gpu} --fold ${fold}
}

train model016 0
train model016 1
train model016 2
train model016 3

