Topic:          Sentiment Analysis, Network Analysis   
Case:           Tweets, Retweets, and Entrepreneurship (QA-0838)
Case Data:      twitter_train.csv 
Network Data:   TechnologyMindzEdges.csv, ChrisGrant360Edges.csv
Code:           TwitterStarterNLPII.ipynb, TwitterStarterNetworkAnalysis.ipynb
Instructions: 	Using Gephi to Visualize a Network

Assignment:
1.	Read the case.
2.	Use the NLP starter code to fit a regression tree that explains the drivers of retweet count in the twitter_train.csv data. Which features are most important? You can run through this code quickly. The code follows the steps from our previous class. Running through it will be a good refresher for those who did not internalize all of the steps the first time through.
3.	One important feature you will find is the user’s followers count. This feature begins to take into account the network effect involved in how a tweet or retweet spreads through the Twitterverse. To consider the network effect in more detail, we have selected two users from the twitter_train.csv file: TechnologyMindz and ChrisGrant360. For these users, we ran the network-analysis starter code to find a list of the user’s followers and the followers of the user’s followers. Take a brief look at the network-analysis starter code, but do not run it. It takes several hours to run because of Twitter’s rate limits.
4.	Use Gephi to visualize these two networks – the one network for the user TechnologyMindz and the other network for the user ChrisGrant360. Gephi is a free, open-source software package for visualizing networks. Please follow the instructions in the “Using Gephi to Visualize a Network” document. How do these two users’ networks differ?
5.	In light of your analyses, what recommendations do you have for TomTom’s Twitter strategy? What types of language should the festival use in its tweets? If the festival were to hire an influencer, what should they look for in his or her network?

Note on Gephi: When trying to install Gephi, if you are getting the error "Cannot find Java 1.8 or higher". The problem is that Gephi can't find where java is installed. To fix this do the following:
1. Go to "C:\Program Files\Gephi-0.9.2\etc" in your finder window
2. Open the file called "gephi.conf"
3. Replace the line '#jdkhome="/path/to/jdk"' (the original has the double quotes, not the single quotes) with the following line 'jdkhome="C:\Program Files (x86)\Java\jre1.8.0_191"' (again only the stuff inside of the single quotes)
4. If the above doesn't work, confirm that "C:\Program Files (x86)\Java\jre1.8.0_191" is the path to your java directory by going to C, then Program Files (x86), then the Java folder, then whatever folder is under there. If you have a different version of java, you may see that the last three numbers of jre1.8.0_191 are different. Replace those numbers with whatever you have.

