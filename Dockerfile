# Dockerfile
FROM python:3.10.12

## Set working directory
WORKDIR /app

## Set the timezone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy files
COPY . .

# Clone the Bert repository
RUN wget https://huggingface.co/microsoft/wavlm-base-plus/resolve/main/pytorch_model.bin?download=true -O slm/wavlm-base-plus/pytorch_model.bin && \
    wget https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm/resolve/main/pytorch_model.bin?download=true -O bert/deberta-v2-large-japanese-char-wwm/pytorch_model.bin && \
    wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin?download=true -O bert/chinese-roberta-wwm-ext-large/pytorch_model.bin && \
    wget https://huggingface.co/microsoft/deberta-v3-large/resolve/main/pytorch_model.bin?download=true -O bert/deberta-v3-large/pytorch_model.bin && \
    wget https://huggingface.co/microsoft/deberta-v3-large/resolve/main/spm.model?download=true -O bert/deberta-v3-large/spm.model && \
    git clone --depth 1 https://huggingface.co/laion/clap-htsat-fused emotional/clap-htsat-fused && \
    git clone --depth 1 https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim


# Install Python requirements
RUN pip install -r requirements.txt

# Set Gradio server name
ENV GRADIO_SERVER_NAME=0.0.0.0

RUN chmod 777 /usr
RUN chmod 777 /app

RUN wget https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz -O /usr/local/lib/python3.10/site-packages/pyopenjtalk/dic.tar.gz
RUN chmod 777 /usr/local/lib/python3.10/site-packages/pyopenjtalk/dic.tar.gz
RUN chmod 777 /usr/local/lib/python3.10/site-packages/pyopenjtalk


RUN mkdir /nltk_data && \
    chmod 777 /nltk_data && \
    mkdir /temp && \
    chmod 777 /temp && \
    mkdir /temp/matplotlib && \
    mkdir /temp/huggingface && \
    mkdir /temp/numba

ENV NUMBA_CACHE_DIR=/temp/numba
ENV MPLCONFIGDIR=/temp/matplotlib
ENV HF_HOME=/temp/huggingface
ENV HOME=/app

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
