gpu=0

test() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python3 ./src/3d_cnn/main.py test ${conf} --gpu ${gpu}
}

test model001
