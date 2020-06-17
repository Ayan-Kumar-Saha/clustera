from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

class Clustera_KMeans:

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto'):

        self.model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X, sample_weight=None):

        fitted_estimator = self.model.fit(X, sample_weight=sample_weight)

        self.labels_ = fitted_estimator.labels_
        self.cluster_centers_ = fitted_estimator.cluster_centers_
        self.inertia_ = fitted_estimator.inertia_
        self.n_iter_ = fitted_estimator.n_iter_

        return fitted_estimator
    
    def fit_predict(self, X, sample_weight=None):   

        return self.model.fit(X, sample_weight=sample_weight).labels_

    def predict(self, X, sample_weight=None):

        return self.model.predict(X, sample_weight=sample_weight)
        
    def accuracy(self, dataset_obj):

        y_pred = self.labels_

        df = dataset_obj.as_dataframe()
        y_true_string = df[dataset_obj.class_attribute_name]

        actual_ALL_count = dataset_obj.class_label_count['ALL']
        actual_AML_count = dataset_obj.class_label_count['AML']

        One_to_ALL = 0   # increases when predicted label = 1 and actual label is ALL
        One_to_AML = 0  # increases when predicted label = 1 and actual label is AML

        Zero_to_ALL = 0  # increases when predicted label = 0 and actual label is ALL
        Zero_to_AML = 0  # increases when predicted label = 0 and actual label is AML

        print('\nCalculating sample counts to every mapping group...')

        for position in range(len(y_pred)):

            if y_pred[position] == 1:

                if y_true_string[position] == 'ALL':
                    One_to_ALL += 1

                else:
                    One_to_AML += 1

            if y_pred[position] == 0:

                if y_true_string[position] == 'ALL':
                    Zero_to_ALL += 1

                else:
                    Zero_to_AML += 1

        # Mapping result
        print('\nShowing mapping results...')

        print('\nMapping result/count:')
        print('------------------------\n')

        One_to_ALL_percentage = (One_to_ALL/actual_ALL_count)*100
        Zero_to_ALL_percentage = (Zero_to_ALL/actual_ALL_count)*100
        One_to_AML_percentage = (One_to_AML/actual_AML_count)*100
        Zero_to_AML_percentage = (Zero_to_AML/actual_AML_count)*100

        print(f'1 mapped to ALL: {One_to_ALL}, Percentage: {One_to_ALL_percentage}')
        print(f'0 mapped to ALL: {Zero_to_ALL}, Percentage: {Zero_to_ALL_percentage}\n')

        print(f'1 mapped to AML: {One_to_AML}, Percentage: {One_to_AML_percentage}')
        print(f'0 mapped to AML: {Zero_to_AML}, Percentage: {Zero_to_AML_percentage}\n\n')

        print('Calculated class labels')
        print('----------------------\n')


        if One_to_ALL_percentage > Zero_to_ALL_percentage:
            calculated_ALL_label = 1
            print('Class 1 is Class ALL')

        else:
            calculated_ALL_label = 0
            print('Class 0 is Class ALL')

        if One_to_AML_percentage > Zero_to_AML_percentage:
            calculated_AML_label = 1
            print('Class 1 is Class AML')

        else:
            calculated_AML_label = 0
            print('Class 0 is Class AML')

        print('\n\nChecking class label conflict...')

        if calculated_ALL_label == calculated_AML_label:
            print('\nConflict found!!')

            print('\nResolving class label conflict...')


            print('\nNew Class labels')
            print('--------------------\n')

            if calculated_ALL_label == 1:
                if One_to_ALL_percentage > One_to_AML_percentage:
                    print('Class 1 is Class ALL')
                    print('Class 0 is Class AML')
                    calculated_AML_label = 0
                    
                else:
                    print('Class 1 is Class AML')
                    print('Class 0 is Class ALL')
                    calculated_ALL_label = 0

            if calculated_ALL_label == 0:
                if Zero_to_ALL_percentage > Zero_to_AML_percentage:
                    print('Class 1 is Class AML')
                    print('Class 0 is Class ALL')
                    calculated_AML_label = 1

                else:
                    print('Class 1 is Class ALL')
                    print('Class 0 is Class AML')
                    calculated_ALL_label = 1
        else:
            print('\nNo conflict found!!')

        y_true = y_true_string.apply(lambda x: calculated_ALL_label if x == 'ALL' else calculated_AML_label)

        print('\nCalculating final results...\n')

        print('\nDisplaying confusion matrix: \n')
        print(confusion_matrix(y_true, y_pred))

        print('\n\nDisplaying classification report: \n')
        print(classification_report(y_true, y_pred))