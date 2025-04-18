# Fraud Transaction Detection 

## 🚀 Project Overview
### Fraud Transaction Prediction is a machine learning project that predicts a transaction is froud or not based on various input features. 

## 🛠 Setup Instructions
To get started, clone the repository and run the following command to set up the environment:

```sh
make setup
```

This will:
- Create a virtual environment (`venv`) if it doesn’t exist.
- Install all required dependencies from `requirements/requirements.txt`.
- Install test dependencies from `requirements/test_requirements.txt`.
- Install the package in editable mode.

### 🏃 Activating Virtual Environment
After setup, activate the virtual environment manually:

```sh
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

## ✅ Running Tests
To run tests, use:

```sh
make test
```

This will execute all tests using `pytest`.

## 🎯 Training the Model
To train the model, run:

```sh
make train
```

This will execute the training pipeline and save the trained model under `fraud_transaction_detection/trained_models/`.

## 🏗️ **Building the Package**
To build the package, use:

```sh
make build
```

This command will:  
- Build the Python package using `python -m build`  
- Verify if the `.whl` file is generated and copy it to the `fraud_transaction_detection_api/` folder  

Ensure you have Python and `pip` installed before running this command.  

---

## 🚀 **Running the Application**
To start the fraud transaction detection API, run:

```sh
make run-app
```

This command will:  
- Start the application on http://0.0.0.0:8001/ 

Ensure the virtual environment (`venv`) is set up before running this command.  

---

## 🐳 **Deploying with Docker**
To build and run the application inside a Docker container, use:

```sh
make docker-deploy
```

This command will:  
- Check if Docker is running  
- Build a Docker image using the `Dockerfile` inside `fraud_transaction_detection_api/`  
- Run the container with the built image
  
Ensure the `.whl` file is generated and copy it to the `fraud_transaction_detection_api/` folder 
Ensure Docker is installed and running before executing this command.  

---



## 💡 Additional Information
- Ensure Python 3.x is installed before running `make setup`.
- If you encounter any issues, try running commands inside the virtual environment manually.

🚀 Happy coding!
