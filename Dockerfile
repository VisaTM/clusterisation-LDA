FROM python:3.7.2
MAINTAINER Jimmy Falck <jimmy.falck@inist.fr>

ENV http_proxy http://proxyout.inist.fr:8080
ENV https_proxy http://proxyout.inist.fr:8080

COPY ./* /usr/bin/
RUN mv /usr/bin/main.py /usr/bin/main
RUN chmod 0755 /usr/bin/
RUN pip install --upgrade pip &&\
    pip3 install numpy &&\
    pip3 install pandas &&\
    pip3 install pyLDAvis &&\
    pip3 install matplotlib &&\
    pip3 install gensim &&\
    pip3 install wordcloud

# ENTRYPOINT ["/usr/bin/main"]
CMD ["main"]
