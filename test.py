# Purpose - Receive the call for testing a page from the Chrome extension and return the result (SAFE/PHISHING)
# for display. This file calls all the different components of the project (The ML model, features_extraction) and
# consolidates the result.

import chardet
import joblib
import features_extraction
import sys
import numpy as np

from features_extraction import LOCALHOST_PATH, DIRECTORY_NAME


def get_prediction_from_url(test_url):
    features_test = features_extraction.main(test_url)
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))

    clf = joblib.load(LOCALHOST_PATH + DIRECTORY_NAME + '\\classifier\\random_forest.pkl')

    pred = clf.predict(features_test)
    return int(pred[0])


def main():
    # url = sys.argv[1]
    f=open("data.txt","r")
    # g1=open("PhishLinks.txt","r")
    # url_phish_list=g1.read()
    with open("PhishLinks.txt", "rb") as file:
            raw_data = file.read()
    detected_encoding = chardet.detect(raw_data)['encoding']
    url_phish_list = raw_data.decode(detected_encoding)
    # print(url_phish_list)
    url=f.read()
    f.close()
    print(url)
    try:
        prediction = get_prediction_from_url(url)
    except:
         prediction = -1
    # Print the probability of prediction (if needed)
    # prob = clf.predict_proba(features_test)
    # print 'Features=', features_test, 'The predicted probability is - ', prob, 'The predicted label is - ', pred
    #    print "The probability of this site being a phishing website is ", features_test[0]*100, "%"
    
    if url in url_phish_list:
        prediction = -1
        # print("PHISHING")
        # exit()

    if prediction == 1:
        # print "The website is safe to browse"
        print("SAFE")
    elif prediction == -1:
        # print "The website has phishing features. DO NOT VISIT!"
        print("PHISHING")
    return prediction

        # print 'Error -', features_test


# if __name__ == "__main__":
#     main()
