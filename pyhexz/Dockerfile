FROM continuumio/miniconda3:latest

# Only copy files necessary to create the conda environment (pytorch and friends).
COPY ./environment.yml /app/

WORKDIR /app

# Create the conda environment. This will download all required Python modules
RUN conda env create -f /app/environment.yml

# Make sure conda is activated in the subsequent RUN commands
RUN echo "conda activate pyhexz" >> ~/.bashrc

SHELL ["/bin/bash", "-c"] 

# Now copy all remaining files.
COPY . ./

RUN source ~/.bashrc && python3 setup.py build_ext --build-lib=src

RUN chmod +x ./entrypoint.sh

# The code to run when container is started:
ENTRYPOINT ["./entrypoint.sh"]

# EXPOSE 8080
