# ARAGORN-ranker

The ARAGORN ranker used by the ARAGORN ARA, which takes a TRAPI 1.1 message containing answers, and calculates numerical scores for each answer.

This is a upgraded port of code from robokop-messenger to perform omnicorp overlay, score and weight-correctness answer ranking.

ARAGORN-ranker uses the omnicorp database to retrieve ontologies. Please see the link below that references that codebase.

## Related Source Code
Below you will find references that detail the standards, web services and supporting tools that are part of ARAGORN. 

* [ARAGORN](https://github.com/ranking-agent/aragorn)
* [Omnicorp](https://github.com/NCATS-Gamma/omnicorp)

### Installation

To run the web server directly:

#### Create a virtual Environment and activate.

    cd <aragorn-ranker root>

    python<version> -m venv venv
    source venv/bin/activate
    
#### Install dependencies

    pip install -r requirements.txt

#### Run Script
  
    cd <aragorn-ranker root>

    ./main.sh
    
 ### DOCKER 
   Or build an image and run it.

    cd <aragorn-ranker root>

    docker build --tag <image_tag> .

   Then start the container

    docker run --name aragorn-ranker -p 8080:4868 aragorn-test

### Kubernetes configurations

Kubernetes configurations and helm charts for this project can be found at: 

    https://github.com/helxplatform/translator-devops/helm/aragorn-ranker
