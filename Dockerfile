FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app



COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app
COPY --chown=user:user extract_roi.py /opt/app
COPY --chown=user:user load_models.py /opt/app
COPY --chown=user:user predict_task1.py /opt/app
COPY --chown=user:user predict_task2.py /opt/app
COPY --chown=user:user checkpoints /opt/app/checkpoints
COPY --chown=user:user swinv2_tiny_model /opt/app/swinv2_tiny_model
COPY --chown=user:user swinv2_scratch_model /opt/app/swinv2_scratch_model
COPY --chown=user:user yolo_model_weights /opt/app/yolo_model_weights




ENTRYPOINT ["python", "inference.py"]
