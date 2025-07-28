
#!/bin/bash

# Parse optional -p flag for pushing images
PUSH=false
while getopts ":p" opt; do
    case $opt in
        p)
            PUSH=true
            ;;
    esac
done
shift $((OPTIND -1))

if [ $# -ne 1 ]; then
    echo "incorrect number of arguments"
    exit 1
fi
VERSION="$1"; shift

IMAGE_NAME="landontclipp/serverless-test"

docker build -t $IMAGE_NAME:$VERSION --platform linux/amd64 .
docker build -t $IMAGE_NAME:latest --platform linux/amd64 .

if [ "$PUSH" = true ]; then
    docker push $IMAGE_NAME:$VERSION
    docker push $IMAGE_NAME:latest
fi