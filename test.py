import cPickle
import gzip
import time
from sklearn.metrics import accuracy_score
from tree import Node, Tree
from boost import Boosting

def load_data():
    data_path = "data/mnist.pkl.gz"
    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data()

X, y = training_data[0], training_data[1]
X_v, y_v = validation_data[0], validation_data[1]
X_t, y_t = test_data[0], test_data[1]

# Test before split entropy calculated when init
node = Node(X_t[:100, :], y_t[:100], verbose=True)
assert node.before_split_entropy is not None

# Test choose best attr
t1 = time.time()
node.choose_best_attr()
t2 = time.time()
print "time: %s" % (t2 - t1)

assert node.best_attr_index is not None
assert node.best_threshold is not None

print node.best_attr_index, node.best_threshold

# Test tree generation
indices = [i for i in np.random.choice(X.shape[0], 5000)]
X_tree = np.array([X[i, :] for i in indices])
y_tree = np.array([y[i] for i in indices])

t1 = time.time()
tree = Tree(X_tree, y_tree)
t2 = time.time()
print "time: %s" % (t2 - t1)
predictions = tree.predict(X_v)
accuracy_score(y_v, predictions)

# Test boost
boosting = Boosting(X, y, n_estimators=128, n_samples=2048)
boosting.train()
predictions = boosting.predict(X_v)
accuracy_score(y_v, predictions)
