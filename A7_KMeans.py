import torch
import numpy as np
import matplotlib.pyplot as plt

N = 10
std = 0.5
k = 2
torch.manual_seed(1)
x = torch.cat((std*torch.randn(2,N)+torch.Tensor([[2],[-2]]), std*torch.randn(2,N)+torch.Tensor([[-2],[2]])),1)

def euclidian(a, b):
        return np.linalg.norm(a-b)

def Plot(c):
    plt.plot(x[0,:N].numpy(), x[1,:N].numpy(), 'ro')
    plt.plot(x[0,N:].numpy(), x[1,N:].numpy(), 'bo')
    l = plt.plot(c[0,:].numpy(), c[1,:].numpy(), 'kx')
    plt.setp(l, markersize=10)
    plt.show()
    
dist_method = euclidian
#num_instances returns number of rows; num_features returns number of columns
num_features, num_instances = x.shape
c = torch.Tensor([[2, -2],[2, -2]])
ctmp = c.transpose(0,1).view(2,2,1)
c_old = np.zeros(c.shape)
print("numInstances " , num_instances , " num_features " , num_features)
belongs_to = np.zeros((num_instances,1))
#Plot(c)
print("x ", x)
print("c " , c)

c0_old = torch.tensor(np.array([[c_old[0,0]],[c_old[1,0]]]))
c0 = torch.tensor(np.array([[c[0,0]],[c[1,0]]]))
c1_old = torch.tensor(np.array([[c_old[0,1]],[c_old[1,1]]]))
c1 = torch.tensor(np.array([[c[0,1]],[c[1,1]]]))
norm = dist_method(c0,c0_old) + dist_method(c1,c1_old)
print("dist0 ", dist_method(c0,c0_old))
print("dist1 ", dist_method(c1,c1_old))
print("norm ", norm)
#for iter in range(num_instances):
for iter in range(10):
    ##############################
    ## compute the distance between points and cluster centers
    ## Dimensions: dist (2x20)
    ##############################

    dist = dist_method(c,c_old)
 #   print("iter ", iter, " dist ", dist)
    #for each datapoint
    c_old = c.clone()
    for index_data in range(20):
#         print("index " , index_data) 
#         print("index_data " , x[0,index_data].numpy())
 #        print("index_data " , x[1,index_data].numpy())
         xVal = x[0,index_data].numpy()
         yVal = x[1,index_data].numpy()
         data = torch.tensor(np.array([[xVal],[yVal]]))
  #       print ("data " , data)
         #initialize distance vector of length k
         dist_vec = np.zeros((k,1))
         #for each centroid
         for index_centroid in range(2):
                 xValClust = c[0,index_centroid].numpy()                 
                 yValClust = c[1,index_centroid].numpy()
                 centroid = torch.tensor(np.array([[xValClust],[yValClust]]))
               #  print ("index_centroid " , index_centroid, " xvalClust ", xValClust, " yvalClust " , yValClust, " centroid " , centroid )
                 #find distance between x and centroid
                 dist_vec[index_centroid] = dist_method(centroid,data)
#                 print("centroid index",index_centroid, " centroid_mean " , centroid, " data ", data, " distance ", dist_vec[index_centroid])
         #find smallest distance and assign to cluster
         belongs_to[index_data, 0] = np.argmin(dist_vec)
    tmp_c = np.zeros((k, num_features))

    #for each cluster
    for index_centroid in range(2):
        #get points assigned to each cluster
         instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index_centroid]
#         print("instances close ", instances_close)
 #        print("x ",x)
         cluster = []
         for index_data in range(20):
                 xVal = x[0,index_data].numpy()
                 yVal = x[1,index_data].numpy()
                 data = torch.tensor(np.array([[xVal],[yVal]]))
                 if index_data in instances_close:
  #                       print("match on index " , index_data)
                         cluster.append(data.tolist())
         print("cluster " , cluster )
        
         mean = np.mean(cluster,axis=0)
  #       print("mean " , mean)
#         print("mean0", mean[0,0])
#         print("mean1", mean[1,0])
#         print("c " , c)
         c[0,index_centroid] = mean[0,0]
#         print("c " , c)
         c[1,index_centroid] = mean[1,0]
#         c = np.reshape(c,c.shape[:-1])
         #add our new centroid to our new temporary list
#         print("c " , c)
#         print("tmp_c ", tmp_c)
         print("iter# ", iter, " loss " , dist_method(c,c_old), " centroids " , c, " c_old " , c_old)
#         tmp_c[index_centroid, :] = c
 #        print("tmp_c after ", tmp_c)
 #   c = tmp_c
    print("iter# ", iter, " loss " , dist_method(c,c_old), " centroids " , c, " c_old " , c_old)
    c0_old = torch.tensor(np.array([[c_old[0,0]],[c_old[1,0]]]))
    c0 = torch.tensor(np.array([[c[0,0]],[c[1,0]]]))
    c1_old = torch.tensor(np.array([[c_old[0,1]],[c_old[1,1]]]))
    c1 = torch.tensor(np.array([[c[0,1]],[c[1,1]]]))
    norm = dist_method(c0,c0_old) + dist_method(c1,c1_old)
 #   print("dist0 ", dist_method(c0,c0_old))
#    print("dist1 ", dist_method(c1,c1_old))
    print("norm ", norm)
    # val,assign = dist.min(0)
    # print("Cost: %f" % torch.sum(val))
    # for k in range(ctmp.size()[0]):
    #     mn = torch.mean(x[:,assign==k],1)
    #     ctmp[k,:,:] = mn.view(-1,1)
    
    Plot(c)

print(ctmp)
