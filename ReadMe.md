# Baby Cry Categorization

This repository contains the necessary files for configuring and running the Baby Cry Categorization application using Docker Compose.

## Prerequisites

Before continuing, make sure the following components are installed on your system:

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## Installation and Configuration

To get started, open a terminal or command prompt and follow the steps below:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/florschmitt/baby_cry_analyzer.git
    ```
2. **Navigate to the cloned folder:**
    ```sh
    cd baby_cry_analyzer
    ```
3. **Copy the "config.yaml" file to the "core" folder:**
    ```sh
    cp path/to/configurations.yaml ./api/core
    ```
4. **Build the Docker images and run the containers:**
    ```sh
    docker-compose build
    docker-compose up
    ```

This will clone the repository, go to the cloned folder, copy the "configurations.yaml" file to the "core" folder, build the Docker images, and run the containers.

## Accessing the Application

After starting the containers, you can access the Baby Cry Categorization application by opening a web browser and navigating to [http://localhost:8000](http://localhost:8000) or the corresponding IP address.

## Stopping the Application

To stop the running containers and remove associated resources, execute the following command from the repository directory:

```sh
docker-compose down
```
