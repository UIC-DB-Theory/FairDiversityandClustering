import fdmalgs
import utils

if __name__ == '__main__':
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
    feature_fields = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt', 'education-num']

    # variables for running LP bin-search
    # keys are appended using underscores
    kis = {
        'White_Male': 15,
        'White_Female': 35,
        'Asian-Pac-Islander_Male': 55,
        'Asian-Pac-Islander_Female': 35,
        'Amer-Indian-Eskimo_Male': 15,
        'Amer-Indian-Eskimo_Female': 35,
        'Other_Male': 15,
        'Other_Female': 35,
        'Black_Male': 15,
        'Black_Female': 35,
    }
    c = len(kis)
    k = sum(kis.values())

    dataset = "./datasets/ads/adult.data"
    colors, features = utils.read_CSV(dataset, allFields, color_field, '_', feature_fields)
    color_counts = {}
    for i in colors:
        if i not in color_counts:
            color_counts[i] = 1
        else:
            color_counts[i] += 1
    assert (len(colors) == len(features))

    N = len(features)

    print(f'***********Parameters***********')
    print(f'Dataset: {dataset}')
    print(f'Result set size(k): {k}')
    print(f'Colors/Grouped By: {color_field}')
    print(f'Num colors(c): {c}')
    print(f'\t\tconstraint\t\tavailable\t\t\tcolor')
    j = 1
    for i in kis:
        print(f'\t{j}\t{kis[i]}\t\t{color_counts[i]}\t\t\t{i}')
        j = j + 1
    print(f'********************************')
    

    print("***********Running FiarFlow on normalized dataset***********")
    #sol1, div_sol1, t1 = algo.FairFlow(X=elements_normalized, k=kis_list, m=c,dist=utils.euclidean_dist)
    sol1, div_sol1, t1 = fdmalgs.FairFlowWrapped(features, colors, kis, normalize=True)
    print("***********Running FiarFlow on non normalized dataset***********")
    #sol2, div_sol2, t2 = algo.FairFlow(X=elements, k=kis_list, m=c,dist=utils.euclidean_dist)
    sol2, div_sol2, t2 = fdmalgs.FairFlowWrapped(features, colors, kis, normalize=False)

    print("\n\n***********Results***********")
    print(f'Solution diversity (normalized) = {div_sol1}')
    print(f'Time taken (normalized) = {t1}')
    print(f'Solution diversity = {div_sol2}')
    print(f'Time taken = {t2}')