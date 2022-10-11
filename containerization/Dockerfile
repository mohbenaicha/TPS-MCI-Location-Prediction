FROM python:3.9.5

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/mci_api


ADD ./mci_api /opt/mci_api

RUN apt-get install -y make && \
 	pip install --upgrade pip

RUN pip install -r /opt/mci_api/requirements.txt

# execute and ownership permssions
# RUN chmod +x /opt/mci_api/run_app.sh
# RUN chown -R api-user:api-user .

EXPOSE 8001
USER api-user