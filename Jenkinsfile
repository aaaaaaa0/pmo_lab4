pipeline {
    agent any
    stages {
        stage('Clone repository') {
            steps {
                checkout scm
            }
        }
        stage('Setup environment') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        stage('Download data') {
            steps {
                sh '''
                    . venv/bin/activate
                    python download.py
                '''
            }
        }
        stage('Train model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python train_model.py
                '''
            }
        }
        stage('Deploy API') {
            steps {
                sh '''
                    . venv/bin/activate
                    export BUILD_ID=dontKillMe
                    export JENKINS_NODE_COOKIE=dontKillMe
                    nohup uvicorn app:app --host 0.0.0.0 --port 5003 > uvicorn.log 2>&1 &
                '''
            }
        }
        stage('Health check') {
            steps {
                sh '''
                    sleep 5
                    curl -X POST http://127.0.0.1:5003/predict \
                        -H "Content-Type: application/json" \
                        -d '{
                            "PassengerId": 999,
                            "Pclass": 1,
                            "Name": "Test, Mr. John",
                            "Sex": "male",
                            "Age": 30.0,
                            "SibSp": 0,
                            "Parch": 0,
                            "Ticket": "ABC123",
                            "Fare": 32.0,
                            "Cabin": null,
                            "Embarked": "S"
                        }'
                '''
            }
        }
    }
}