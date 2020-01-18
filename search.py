# Run the indexing.py first
# $ python search.py --dataset dataset --index data/index

# import the necessary packages
from searcher import Searcher
import numpy as np
import argparse
import pickle as cPickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
                help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
                help = "Path to where we stored our index")
args = vars(ap.parse_args())

# load the index and initialize our 
#file = open(filename, encoding="utf8")
index = cPickle.loads(open(args["index"],'rb').read())
searcher = Searcher(index)

# loop over images in the index -- we will use each one as
# a query image
for (query, queryFeatures) in index.items():
    # perform the search using the current query
    results = searcher.search(queryFeatures)

    # load the query image and display it
    #path = args["dataset"] + "/%s" % (query)
    path = query
    queryImage = cv2.imread(path)
    cv2.imshow("Query", queryImage)

    # initialize the two montages to display our results --
    # we have a total of 12 images in the index, but let's only
    # display the top 10 results; 5 images per montage, with
    # images that are 400x166 pixels
    montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
    montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")

    # loop over the top ten results
    for j in range(0, 10):
        # grab the result (we are using row-major order) and
        # load the result image
        (score, imageName) = results[j]
        #path = args["dataset"] + "/%s" % (imageName)
        path = imageName
        result = cv2.imread(path)
		
        print("Title: ", imageName, "score: ", str(score))
		
        # start : stop : step
        #montageA[0:166, :] = result
		
        # resize result
        result = cv2.resize(result, (400,166))

        if j < 5:
            montageA[j * 166:(j + 1) * 166, :] = result
            #cv2.imshow("Results 1-5", result)
            #cv2.imshow("Montage 1-5", montageA)
            #cv2.waitKey(0)
        else:
            montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result
            #cv2.imshow("Results 6-10", result)

            # Show the result
            cv2.imshow("Montage 1-5", montageA)
            cv2.imshow("Montage 6-10", montageB)
            cv2.waitKey(0)
'''		
		#print "\t%d. %s : %.3f" % (j + 1, imageName, score)

        # check to see if the first montage should be used
        if j < 5:
            #montageA[j * 166:(j + 1) * 166, :] = result
            

            # otherwise, the second montage should be used
        else:
            montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result

            # show the results
            cv2.imshow("Results 1-5", montageA)
            cv2.imshow("Results 6-10", montageB)
            cv2.waitKey(0)
'''

    
