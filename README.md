# Anomaly-Detection-in-MachineLearning
This repository contains a few methods of detecting Anomaly in a dataset using Scikit Learn
Anomaly_Detection.ipynb
In 'Anomaly_Detection.ipynb' file I have taken a Credit cArd Dataset to find anomalies in the dataset.Here i have used Isolation forest method and Gaussian model to predict the anomalies. There is some in finding anomalies in this model as the dataset was not appropriate for the detection of anomalies.
I have learned a lot while building this model in(Anomaly_Detection.ipynb)

In 'AnomalyDetection_on_Cpu_Utlization.ipynb' I have taken a dataset of Cpu utilization values. Here inthe dataset it contains the value of cpu utilization in different time periods. There shall be anomalies in certain time Period. Those Anomalies are found using Isolation_forest Method.
This model have found the anomalies in a better way than the preious model.

                                         #Isolation Forest Method
Isolation Forest is an anomaly detection algorithm that works by isolating observations in the dataset. The basic idea is that anomalies are 'few and different' and thus easier to isolate than normal observations. Isolation Forest builds an ensemble of decision trees, and the path length from the root to a leaf is used to determine the anomaly score of an observation. Shorter paths are more likely to correspond to anomalies.
                                       #Steps in Isolation Forest Method
1. Each IsolationTree(iTree) is constructed by recursively partitioning the data space. At each node, a random feature 𝑓 is selected and a 
   random split value 𝑣 is chosen between the minimum and maximum values of the selected feature.
2. The path length ℎ(𝑥)of a point x is the number of edges from the root node to the leaf node in the isolationTree.
3. The anomaly score 
          s(x,n) is then calculated as: 𝑠(𝑥,𝑛)=2^(−𝐸(ℎ(𝑥))𝑐(𝑛))

4. The anomaly score ranges from 0 to 1.
   A score close to 1 indicates a high likelihood of 𝑥 being an anomaly.
   A score much smaller than 0.5 indicates x is likely a normal point.
   Scores around 0.5 are ambiguous.

​
 
 
