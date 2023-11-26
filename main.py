import yaml
from knn import knn
from k_means import k_means
from confusion_matrix import confusion_matrix_cal

if __name__ == "__main__":
    with open('questions.yml', 'r') as file:
        data = yaml.safe_load(file)

    while True:
        print('\n\nSoru listesi:')
        for idx,key in enumerate(data):
            print(f'{idx+1}: {key}')
        print('q: quit')
        question = input('Please enter the question number: ')
        if question == 'q':
            break

        if question == '1':
            if 'knn' in data:
                k = data['knn']['k']
                quiz = data['knn']['quiz']
                knn(data, k, quiz)
            else :
                print('Knn sorusu bulunamadı.')
        elif question == '2':
            if 'k-means' in data:
                values = data['k-means']['values']
                initial_centroids = data['k-means']['cendroids']
                k = len(initial_centroids)
                k_means(values, initial_centroids, k)
            else :
                print('K-means sorusu bulunamadı.')
        elif question == '3':
            if 'confusion_matrix' in data:
                y_true = data['confusion_matrix']['actual']
                y_pred = data['confusion_matrix']['predicted']
                confusion_matrix_cal(y_true, y_pred)
            else :
                print('Confusion matrix sorusu bulunamadı.')