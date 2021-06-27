FROM nvcr.io/nvidia/clara-train-sdk:v4.0

# Update
RUN apt-get update -y
RUN apt-get upgrade -y

# Adding all repository into the container
WORKDIR /home/raian/Documents/repos/pectoral-muscle-segmentation
# WORKDIR /home/ec2-user/SageMaker/pectoral-muscle-segmentation

# Installing libraries
ADD ./requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install monai
RUN pip install 'monai[all]'

RUN conda update --all
RUN conda install -c conda-forge scikit-image

# ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]

# NOTE: To run the container you can use this command:
# >>> export dockerImageName=pectoral-muscle-clara-monai
# >>> nvidia-docker build -t $dockerImageName . 
# >>> nvidia-docker run --gpus all \
# >>>                   --ipc=host \
# >>>                   --shm-size=16g \ # --> RAM Memory
# >>>                   --name pectoral-muscle-research \
# >>>                   -it \
# >>>                   --net=host
# >>>                   -v /home/raian/Documents/repos/pectoral-muscle-segmentation/:/home/raian/Documents/repos/pectoral-muscle-segmentation \
# >>>                   $dockerImageName 