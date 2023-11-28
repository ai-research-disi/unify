# Step 1: Choose a base image
FROM ubuntu:18.04

# Step 2: Update the package lists
RUN apt-get update

# Step 3: Install Python 3.8 and pip
RUN apt-get install -y python3.8 python3.8-dev python3-pip

# Step 4: Set the working directory
WORKDIR /app

# Step 5: Copy your Python application
COPY . /app

# Step 6: Install application dependencies
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

# Step 7: Set the Python 3.8 environment as default
RUN update-alternatives --install /usr/bin/python3 python3.8 /usr/bin/python3.8 1
