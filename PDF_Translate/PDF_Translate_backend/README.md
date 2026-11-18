
docker stop my-translate && docker rm my-translate

docker build -t my-translate .

docker run -d --name my-translate -p 8089:8089 my-translate
