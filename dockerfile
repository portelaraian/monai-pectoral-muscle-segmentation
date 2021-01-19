FROM nvcr.io/nvidia/pytorch:20.12-py3

# Update
RUN apt-get update -y
RUN apt-get upgrade -y

# Install JupyterLab
RUN pip install jupyterlab
EXPOSE 8888

# Adding all repository into the container
WORKDIR /home/raian/Documents/repos/

# Installing libraries
ADD ./requirements.txt ./
RUN pip install -r requirements.txt
RUN conda update --all

RUN conda install -c conda-forge scikit-image
RUN pip install git+https://github.com/ncullen93/torchsample
RUN pip install git+https://github.com/ildoonet/pytorch-randaugment
RUN apt install libgl1-mesa-glx -y
RUN apt install default-jre -y


ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
