
from feature1 import X_PCA, y

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,precision_score, recall_score, f1_score


"""

Feature 2 description: 

- Perform a clasiffication model that classifies pacients as malign or benign
- We use a Support Vector Machine supervised learning model to predict that

"""


# Fixed: added test_sample_size and rand_state to let user choose their test split parameters
def prepare_data(X ,y ,test_sample_size ,rand_state):
    """

    Prepare data for our model it splits both the feature matrix and label vector to train and test

    Args:
        X (np.ndarray or pd.DataFrame): feature matrix
        y (np.ndarray or pd.DataFrame): label vector
        rand_state (int): random state for split
        test_sample_size (float68): select percentage of test sample

    Returns:
        tuple: Tuple with len = 4:
        - X_train: training feature matrix
        - X_test: testing feature matrix
        - y_train: training label vector
        - y_test: testing label vector
    """

    X_train ,X_test, y_train ,y_test =train_test_split(X, y, test_size=test_sample_size, random_state=rand_state)
    return X_train, X_test, y_train, y_test


# Added kernel and random_state as a parameter
# Added more metrics for model performance: preciion,recal,f1

