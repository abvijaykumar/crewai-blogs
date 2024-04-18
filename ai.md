In the realm of machine learning, there are three primary types to master: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Let's explore each type in detail with suitable code examples and tutorials using Python and popular libraries like scikit-learn and qlearn.

1. **Supervised Learning:** In supervised learning, we train models on labeled data. The algorithm uses example input-output pairs to learn how to map new inputs to their corresponding outputs. This method allows the model to learn patterns and relationships from the training data. Suitable for regression tasks (predicting continuous values) and classification tasks (predicting discrete labels).

Example: Let's create a simple linear regression model using scikit-learn:

```python
from sklearn import datasets, linear_model

# Load iris dataset
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Create a linear regression model and train it on the data
clf = linear_model.LinearRegression()
clf.fit(X, y)

# Predict the target for new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_data)
print(prediction)
```

2. **Unsupervised Learning:** In unsupervised learning, the model learns patterns and relationships from unlabeled data. The goal is to discover hidden structures within the data, such as clusters or dimensions. Common applications include data compression, anomaly detection, and dimensionality reduction.

Example: Let's examine K-Means clustering, a popular unsupervised learning algorithm:

```python
from sklearn import datasets, cluster
import numpy as np

# Load iris dataset
iris = datasets.load_iris()
X = iris['data']

# Run K-means clustering with 3 clusters
kmeans = cluster.KMeans(n_clusters=3).fit(X)

# Get the cluster labels for new data
new_data = np.array([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4]])
predictions = kmeans.predict(new_data)
print(predictions)
```

3. **Reinforcement Learning:** Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with its environment and receiving feedback in the form of rewards or penalties. The goal is for the agent to maximize its long-term reward by learning which actions lead to desirable outcomes. Suitable for controlling complex systems like robotics and autonomous vehicles.

Example: Let's explore Q-Learning, a popular reinforcement learning algorithm:

```python
import numpy as np
from qlearn import QLearner

# Create an environment with 4 possible actions and 2 states
states = ['S0', 'S1']
actions = ['A0', 'A1', 'A2', 'A3']
env = {'state_space': states, 'action_space': actions}

# Define a simple reward function
rewards = {
    'S0_A0': 1.0, 'S0_A1': -1.0, 'S0_A2': 0.5, 'S0_A3': -0.5,
    'S1_A0': -0.5, 'S1_A1': 1.0, 'S1_A2': -1.0, 'S1_A3': 0.5
}

# Initialize a Q-Learning agent and train it on the environment
ql = QLearner(state_space=states, action_space=actions, rewards=rewards)
ql.fit(num_episodes=1000, visualize=False)
```


In the realm of machine learning, there are three primary types to master: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Let's explore each type in detail with suitable code examples and tutorials using Python and popular libraries like scikit-learn and qlearn.

1. **Supervised Learning:** In supervised learning, we train models on labeled data. The algorithm uses example input-output pairs to learn how to map new inputs to their corresponding outputs. This method allows the model to learn patterns and relationships from the training data. Suitable for regression tasks (predicting continuous values) and classification tasks (predicting discrete labels).

Example: Let's create a simple linear regression model using scikit-learn:

```python
from sklearn import datasets, linear_model

# Load iris dataset
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Create a linear regression model and train it on the data
clf = linear_model.LinearRegression()
clf.fit(X, y)

# Predict the target for new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_data)
print(prediction)
```

2. **Unsupervised Learning:** In unsupervised learning, the model learns patterns and relationships from unlabeled data. The goal is to discover hidden structures within the data, such as clusters or dimensions. Common applications include data compression, anomaly detection, and dimensionality reduction.

Example: Let's examine K-Means clustering, a popular unsupervised learning algorithm:

```python
from sklearn import datasets, cluster
import numpy as np

# Load iris dataset
iris = datasets.load_iris()
X = iris['data']

# Run K-means clustering with 3 clusters
kmeans = cluster.KMeans(n_clusters=3).fit(X)

# Get the cluster labels for new data
new_data = np.array([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4]])
predictions = kmeans.predict(new_data)
print(predictions)
```

3. **Reinforcement Learning:** Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with its environment and receiving feedback in the form of rewards or penalties. The goal is for the agent to maximize its long-term reward by learning which actions lead to desirable outcomes. Suitable for controlling complex systems like robotics and autonomous vehicles.

Example: Let's explore Q-Learning, a popular reinforcement learning algorithm:

```python
import numpy as np
from qlearn import QLearner

# Create an environment with 4 possible actions and 2 states
states = ['S0', 'S1']
actions = ['A0', 'A1', 'A2', 'A3']
env = {'state_space': states, 'action_space': actions}

# Define a simple reward function
rewards = {
    'S0_A0': 1.0, 'S0_A1': -1.0, 'S0_A2': 0.5, 'S0_A3': -0.5,
    'S1_A0': -0.5, 'S1_A1': 1.0, 'S1_A2': -1.0, 'S1_A3': 0.5
}

# Initialize a Q-Learning agent and train it on the environment
ql = QLearner(state_space=states, action_space=actions, rewards=rewards)
ql.fit(num_episodes=1000, visualize=False)