postprocess() {
    model=$1
    
    conf=./conf/${model}.py
    python3 ./src/postprocess/pectoral_major_area_selection.py ${conf} --n_pool 8
}

postprocess model003