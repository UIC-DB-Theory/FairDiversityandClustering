import utils
from WeightedTree import *

# Example usage.
if __name__ == "__main__":
    # setup
    # File fields
    allFields = [
        "age",
        "workclass",
        "fnlwgt",  # what on earth is this one?
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "yearly-income",
    ]

    # fields we care about for parsing
    color_field = ['race', 'sex']
    feature_fields = {'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num'}

    # variables for running LP bin-search

    # coreset params
    # Set the size of the coreset

    colors, features = utils.read_CSV("./datasets/ads/adult.data", allFields, color_field, '_', feature_fields)
    assert (len(colors) == len(features))
    N = len(features)
    h = np.full((N, 1), 1.0 / N, dtype=float)  # weights
    # Initialize the tree and the subprocess.
    tree = WeightedTree(len(feature_fields))
    print(len(feature_fields))
    print(type(features))

    tree_construction_response = tree.construct_tree(features)

    # Construct the tree on the data.
    # tree_construction_response = tree.construct_tree(np.array([0.0, 0.0, 0.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0,
    #                                                           1.0, 2.0, 3.0]))

    print(f"Constructed a tree: {tree_construction_response}")
    #
    # # Run a query on the tree.
    #
    time1, result1 = tree.run_query(1, h)
    # result1 = tree.run_query(1, np.array([1, 1, 1]))
    print(f"Query 1 Result: {result1}")
    h = h / 2
    # # Run a query.
    #
    time2, result2 = tree.run_query(.5, h)
    print(f"Query 2 Result: {result2}")
    #
    # # Delete the tree. Ensures the process exits correctly
    tree.delete_tree()
