FROM swr.ap-southeast-1.myhuaweicloud.com/atelier/mindspore_2_3_ascend:mindspore_2.3.0-cann_8.0.rc2-py_3.9-euler_2.10.7-aarch64-snt9b-20240727152329-0f2c29a
USER root
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

COPY . .

ENV PYTHONPATH="/app"

RUN chmod +x run_job.sh

USER ma-user

ENTRYPOINT ["/bin/bash", "run_job.sh"]