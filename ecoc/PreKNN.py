from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# from sklearn.svm import libsvm
from sklearn import preprocessing
from ecoc.FeatureSelection.FeatureSelector import BSSWSS
# from sklearn.svm import libsvm
import sklearn.svm
# from svmutil import *
from libsvm import svmutil
from libsvm.svmutil import *
import sklearn.metrics as metrics
from ecoc.DataComplexity.datacomplexity import get_data_complexity
from matplotlib import pyplot as plt


class PreKNN:

    def __init__(self, tr_labels, tv_data, tv_labels):
        self.model = None
        self.tr_labels = tr_labels
        self.tv_data = tv_data
        self.tv_labels = tv_labels
        self.tr_data = None
        self.pos_cols_list = []
        self.neg_cols_list = []
        # print('yyyyyyyy')

    # def fit_predict(self,data,labels,test_data):
    #     model=KNeighborsClassifier(n_neighbors=self.performance_matrix.shape[0])
    #     temp_labels=[]
    #     for i in range(data.shape[0]):
    #         temp_labels.append(np.argmax(labels[:, i]))
    #     temp_labels=np.array(temp_labels)
    #     model.fit(data,temp_labels)
    #     temp=model.kneighbors(test_data)
    #     print(temp[1])
    #     self.model=model
    #     self.tr_data=data
    #     score=self.model.score(data,temp_labels)
    #     print(score)
    #     # self.single_decode(test_data[0,:],self.pre_labels[0])

    # def single_decode(self,data,label):
    #     performance_matrix=self.performance_matrix.copy()
    #     performance_row=performance_matrix[label-1,:]
    #     for i in range(len(performance_row)):
    #         performance_flag=performance_row[i]
    #         temp=[]
    #         p_labels, _, _ = svm_predict([], data, self.svm_models[i])
    #         print(p_labels,performance_flag)

    def fit(self, data, labels):
        # model = KNeighborsClassifier(n_neighbors=labels.shape[0]) #n_neighbors=4498? 13个，进来的是partial，第一维是13
        model = KNeighborsClassifier(n_neighbors=13)
        temp_labels = np.zeros((1, data.shape[0])).T #data.shape 4499, 38 shape(4499, 1
        self.labels = labels
        model.fit(data, temp_labels.ravel())
        self.model = model

    def predict(self, pre_data, true_labels):
        neighbors = self.model.kneighbors(pre_data)
        pre_knn_labels_matrix = None
        for i in range(pre_data.shape[0]):
            temp_distances = neighbors[0][i]
            temp_indexs = neighbors[1][i]
            distances_sum = np.sum(temp_distances)
            temp_pre_labels_matrix = np.zeros((1, self.tr_labels.shape[0])).T
            temp_pre_distances_matrix = np.zeros(
                (1, self.tr_labels.shape[0])).T
            temp_pre_weight = np.zeros((1, len(temp_indexs))).T
            for j in range(len(temp_indexs)):
                temp_pre_weight[j] = 1-temp_distances[j]/distances_sum
                for index in np.where(self.tr_labels[:, temp_indexs[j]] == 1)[0]:
                    temp_pre_distances_matrix[index][0] += temp_distances[j]
                    temp_pre_labels_matrix[index][0] += temp_pre_weight[j]
            # print(np.where(temp_pre_labels_matrix==temp_pre_labels_matrix.max()))
            # self.pre_knn_perfomance_matrix=temp_pre_distances_matrix if self.pre_knn_perfomance_matrix is None else np.hstack((self.pre_knn_perfomance_matrix,temp_pre_distances_matrix))
            pre_knn_labels_matrix = temp_pre_labels_matrix if pre_knn_labels_matrix is None else np.hstack(
                (pre_knn_labels_matrix, temp_pre_labels_matrix))
        # self.pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(self.pre_knn_perfomance_matrix)
        # self.pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(self.pre_knn_labels_matrix)
        pre_knn_perfomance_matrix = preprocessing.StandardScaler(
        ).fit_transform(pre_knn_labels_matrix)
        pre_knn_perfomance_matrix = preprocessing.MinMaxScaler(
        ).fit_transform(pre_knn_perfomance_matrix)

        pre_label_matrix = np.zeros(
            (self.tr_labels.shape[0], pre_data.shape[0]))
        for i in range(pre_data.shape[0]):
            idx = pre_knn_labels_matrix[:, i] == max(
                pre_knn_labels_matrix[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(pre_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.where(true_labels[:, i]==1)
            if np.argwhere((max_idx2==max_idx1)==True).shape[0]!=0:
                count = count+1
        knn_accuracy = count / pre_data.shape[0]

        return pre_label_matrix, knn_accuracy, pre_knn_perfomance_matrix

    def getWeight(self,knn_matrix,ecoc_matrix):
        weight=1
        decline=0.05
        decline_rate=0.02
        tmp_accuracy=0
        acc_list=np.zeros(int(weight/decline)+1)
        index=0
        while(weight>=0):
            count=0
            pre_label_matrix=knn_matrix*(1-weight)+ecoc_matrix*weight
            for i in range(pre_label_matrix.shape[1]):
                max_idx1 = np.argmax(pre_label_matrix[:, i])
                max_idx2 = np.where(self.tv_labels[:, i]==1)
                if np.argwhere((max_idx2==max_idx1)==True).shape[0]!=0:
                    count = count+1
            accuracy=count / pre_label_matrix.shape[1]
            # if(tmp_accuracy-accuracy>tmp_accuracy*decline_rate):
            #     break
            acc_list[index]=accuracy
            index+=1
            weight-=decline
            tmp_accuracy=accuracy
        acc_list=acc_list.tolist()
        draw_hist(acc_list," ","times","accuracy",0,len(acc_list),0,1)
        return 1-np.argmax(acc_list)*0.05
        
    def getValidationData(self):
        return self.tv_data, self.tv_labels


class PLFeatureSelection:

    score_list = []

    def __init__(self, tr_data, tr_labels, tv_data, tv_labels, params):
        self.num_features = tr_data.shape[1]
        self.tr_data = tr_data
        self.tr_labels = tr_labels
        self.fs_model = None
        self.tv_data = tv_data
        self.tv_labels = tv_labels
        self.params = params
        self.decline_rate = 0.02

    def matrix_test(self, matrix, tmp_tr_pos_idx, tmp_tr_neg_idx):
        f1_score_list = []
        for i in range(matrix.shape[0]):
            tmp_col = matrix[i, :]
            tr_pos_idx = tmp_tr_pos_idx[i]
            tr_neg_idx = tmp_tr_neg_idx[i]
            col_f1_score = self.col_test(tr_pos_idx, tr_neg_idx, tmp_col)
            f1_score_list.append(col_f1_score)
        f1_score_list=np.array(f1_score_list)
        mean_f1_score=f1_score_list.mean()
        high_score_list=np.where(f1_score_list>mean_f1_score)
        return high_score_list[0]
    def col_test(self, tr_pos_idx, tr_neg_idx, coding_col):#产生正负样本标签然后根据SVM一分为二进行测试
        # coding_col=self.coding_col.tolist()
        # for i in range(data.shape[0]):
        #     temp_labels=np.where(labels[i,:]==1)[0]
        #     num_pos=0
        #     num_neg=0
        #     for class_index in temp_labels:
        #         if(coding_col[class_index]==1):
        #             num_pos+=1
        #         elif(coding_col[class_index]==-1):
        #             num_neg+=1
        #     print(num_pos)
        # print('============================================================================================')
        # print(coding_col)
        # print(np.where(coding_col == 0))
        tv_pos_idx = []
        tv_neg_idx = []
        tv_data_flag = np.zeros(self.tv_data.shape[0])#验证集有多少个
        coding_col[np.where(coding_col == -1)[0]] = 0#为什么是-1 不应该是0吗,懂了，他在以防万一，虽然都是01怕有-1 的
        # coding_col[np.where(coding_col == 0)[0]] = 0
        #     print(num_neg)
        '''
        到这里了
        '''
        for j in range(self.tv_data.shape[0]):#训练样本的个数
            if np.all((self.tv_labels[:, j] & coding_col) == self.tv_labels[:, j]):#这个是验证集或者是测试集的了！！
                tv_pos_idx.append(j)
                tv_data_flag[j] += 1
            else:
                if np.all((self.tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == self.tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j] += 1
        # print(len(np.where(tv_data_flag == 0)[0]))
        pos_inst = self.tv_data[tv_pos_idx]
        neg_inst = self.tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))

        tr_pos_inst = self.tr_data[tr_pos_idx]
        tr_neg_inst = self.tr_data[tr_neg_idx]
        data = np.vstack((tr_pos_inst, tr_neg_inst))
        labels = np.hstack(
            (np.ones(len(tr_pos_inst)), -np.ones(len(tr_neg_inst))))#标签组成是+1 and -1

        # acc_list = np.zeros((self.num_features, 4))
        # fs_model = BSSWSS(k=self.num_features)  # remain 2 features.
        # fs_model.fit(data, labels)

        #.632自助法
        data=np.vstack((data,tv_inst))#训练集+测试集所有的正负数据样本
        labels=np.hstack((labels,tv_labels))
        bootstrapping = []
        bootstrapping_flag=np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            bootstrapping.append(np.floor(np.random.random()*data.shape[0]))
            bootstrapping_flag[int(bootstrapping[i])]=1
        tv_data_index=np.where(bootstrapping_flag==0)[0].tolist()
        tv_inst=data[tv_data_index]
        tv_labels=labels[tv_data_index]
        bootstrapping=np.int8(np.array(bootstrapping))
        data=data[bootstrapping]
        labels=labels[bootstrapping]

        prob = svm_problem(labels.tolist(),
                           data.tolist())
        param = svm_parameter(self.params.get('svm_param'))
        model = svm_train(prob, param)
        tmp_tv_inst = tv_inst
        p1, p2, p3 = svm_predict(
            tv_labels, tmp_tv_inst.tolist(), model)
        
        accuracy = p2[0]
        precision = metrics.precision_score(tv_labels, p1)
        recall = metrics.recall_score(tv_labels, p1)
        f1_score = metrics.f1_score(tv_labels, p1)
        # acc_list[i][0] = accuracy
        # acc_list[i][1] = precision
        # acc_list[i][2] = recall
        # acc_list[i][3] = f1_score
        # if(tmp_f1_score-f1_score>tmp_f1_score*self.decline_rate):
        #     break
        # else:
        #     tmp_f1_score=f1_score
        # print(str(np.argmax(acc_list[:, 0]))+"："+str(acc_list[:, 0].max()))
        # print(str(np.argmax(acc_list[:, 1]))+"："+str(acc_list[:, 1].max()))
        # print(str(np.argmax(acc_list[:, 2]))+"："+str(acc_list[:, 2].max()))
        # print(str(np.argmax(acc_list[:, 3]))+"："+str(acc_list[:, 3].max()))
        # fs_model = BSSWSS(k=self.num_features-np.argmax(acc_list[:, 3]))
        # fs_model.fit(data, labels)
        return f1_score
    def matrix_test_4_1(self, matrix, tmp_tr_pos_idx, tmp_tr_neg_idx):
        f1_score_list = []
        acc_list=[]
        matrix=matrix.T
        for i in range(matrix.shape[0]):
            tmp_col = matrix[i, :]
            tr_pos_idx = tmp_tr_pos_idx[i]
            tr_neg_idx = tmp_tr_neg_idx[i]
            col_f1_score, col_acc= self.col_test_4_1(tr_pos_idx, tr_neg_idx, tmp_col)
            f1_score_list.append(col_f1_score/100)
            acc_list.append(col_acc)
        return f1_score_list,acc_list
        
    def col_test_4_1(self, tr_pos_idx, tr_neg_idx, coding_col):
        # coding_col=self.coding_col.tolist()
        # for i in range(data.shape[0]):
        #     temp_labels=np.where(labels[i,:]==1)[0]
        #     num_pos=0
        #     num_neg=0
        #     for class_index in temp_labels:
        #         if(coding_col[class_index]==1):
        #             num_pos+=1
        #         elif(coding_col[class_index]==-1):
        #             num_neg+=1
        #     print(num_pos)
        tv_pos_idx = []
        tv_neg_idx = []
        tv_data_flag = np.zeros(self.tv_data.shape[0])
        coding_col[np.where(coding_col == -1)[0]] = 0
        #     print(num_neg)
        for j in range(self.tv_data.shape[0]):
            if np.all((self.tv_labels[:, j] & coding_col) == self.tv_labels[:, j]):
                tv_pos_idx.append(j)
                tv_data_flag[j] += 1
            else:
                if np.all((self.tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == self.tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j] += 1
        # print(len(np.where(tv_data_flag == 0)[0]))
        pos_inst = self.tv_data[tv_pos_idx]
        neg_inst = self.tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))

        tr_pos_inst = self.tr_data[tr_pos_idx]
        tr_neg_inst = self.tr_data[tr_neg_idx]
        data = np.vstack((tr_pos_inst, tr_neg_inst))
        labels = np.hstack(
            (np.ones(len(tr_pos_inst)), -np.ones(len(tr_neg_inst))))

        # acc_list = np.zeros((self.num_features, 4))
        # fs_model = BSSWSS(k=self.num_features)  # remain 2 features.
        # fs_model.fit(data, labels)

        #.632自助法
        data=np.vstack((data,tv_inst))
        labels=np.hstack((labels,tv_labels))
        bootstrapping = []
        bootstrapping_flag=np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            bootstrapping.append(np.floor(np.random.random()*data.shape[0]))
            bootstrapping_flag[int(bootstrapping[i])]=1
        tv_data_index=np.where(bootstrapping_flag==0)[0].tolist()
        tv_inst=data[tv_data_index]
        tv_labels=labels[tv_data_index]
        bootstrapping=np.int8(np.array(bootstrapping))
        data=data[bootstrapping]
        labels=labels[bootstrapping]

        prob = svm_problem(labels.tolist(),
                           data.tolist())
        param = svm_parameter(self.params.get('svm_param'))
        model = svm_train(prob, param)
        tmp_tv_inst = tv_inst
        p1, p2, p3 = svm_predict(
            tv_labels, tmp_tv_inst.tolist(), model)
        
        accuracy = p2[0]
        precision = metrics.precision_score(tv_labels, p1)
        recall = metrics.recall_score(tv_labels, p1)
        f1_score = metrics.f1_score(tv_labels, p1)
        # acc_list[i][0] = accuracy
        # acc_list[i][1] = precision
        # acc_list[i][2] = recall
        # acc_list[i][3] = f1_score
        # if(tmp_f1_score-f1_score>tmp_f1_score*self.decline_rate):
        #     break
        # else:
        #     tmp_f1_score=f1_score
        # print(str(np.argmax(acc_list[:, 0]))+"："+str(acc_list[:, 0].max()))
        # print(str(np.argmax(acc_list[:, 1]))+"："+str(acc_list[:, 1].max()))
        # print(str(np.argmax(acc_list[:, 2]))+"："+str(acc_list[:, 2].max()))
        # print(str(np.argmax(acc_list[:, 3]))+"："+str(acc_list[:, 3].max()))
        # fs_model = BSSWSS(k=self.num_features-np.argmax(acc_list[:, 3]))
        # fs_model.fit(data, labels)
        return accuracy,f1_score
    # def col_test(self, tr_pos_idx, tr_neg_idx, coding_col):
        # coding_col=self.coding_col.tolist()
        # for i in range(data.shape[0]):
        #     temp_labels=np.where(labels[i,:]==1)[0]
        #     num_pos=0
        #     num_neg=0
        #     for class_index in temp_labels:
        #         if(coding_col[class_index]==1):
        #             num_pos+=1
        #         elif(coding_col[class_index]==-1):
        #             num_neg+=1
        #     print(num_pos)
        tv_pos_idx = []
        tv_neg_idx = []
        tv_data_flag = np.zeros(self.tv_data.shape[0])
        coding_col[np.where(coding_col == -1)[0]] = 0
        #     print(num_neg)
        for j in range(self.tv_data.shape[0]):
            if np.all((self.tv_labels[:, j] & coding_col) == self.tv_labels[:, j]):
                tv_pos_idx.append(j)
                tv_data_flag[j] += 1
            else:
                if np.all((self.tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == self.tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j] += 1
        print(len(np.where(tv_data_flag == 0)[0]))
        pos_inst = self.tv_data[tv_pos_idx]
        neg_inst = self.tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))

        tr_pos_inst = self.tr_data[tr_pos_idx]
        tr_neg_inst = self.tr_data[tr_neg_idx]
        data = np.vstack((tr_pos_inst, tr_neg_inst))
        labels = np.hstack(
            (np.ones(len(tr_pos_inst)), -np.ones(len(tr_neg_inst))))

        dc = get_data_complexity('F1')
        return dc.score(data, labels)

    def fit(self, data, labels, coding_col):
        # coding_col=self.coding_col.tolist()
        # for i in range(data.shape[0]):
        #     temp_labels=np.where(labels[i,:]==1)[0]
        #     num_pos=0
        #     num_neg=0
        #     for class_index in temp_labels:
        #         if(coding_col[class_index]==1):
        #             num_pos+=1
        #         elif(coding_col[class_index]==-1):
        #             num_neg+=1
        #     print(num_pos)
        tv_data = self.tv_data.copy()
        tv_labels = self.tv_labels.copy()
        tv_pos_idx = []
        tv_neg_idx = []
        tv_data_flag = np.zeros(tv_data.shape[0])
        coding_col[np.where(coding_col == -1)[0]] = 0
        #     print(num_neg)
        for j in range(tv_data.shape[0]):
            if np.all((tv_labels[:, j] & coding_col) == tv_labels[:, j]):
                tv_pos_idx.append(j)
                tv_data_flag[j] += 1
            else:
                if np.all((tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j] += 1
        print(len(np.where(tv_data_flag == 0)[0]))
        pos_inst = tv_data[tv_pos_idx]
        neg_inst = tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
        
        #.632自助法
        data=np.vstack((data,tv_inst))
        labels=np.hstack((labels,tv_labels))
        bootstrapping = []
        bootstrapping_flag=np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            bootstrapping.append(np.floor(np.random.random()*data.shape[0]))
            bootstrapping_flag[int(bootstrapping[i])]=1
        tv_data_index=np.where(bootstrapping_flag==0)[0].tolist()
        tv_inst=data[tv_data_index]
        tv_labels=labels[tv_data_index]
        bootstrapping=np.int8(np.array(bootstrapping))
        data=data[bootstrapping]
        labels=labels[bootstrapping]


        acc_list = np.zeros((self.num_features, 4))
        tmp_f1_score = 0

        for i in range(self.num_features):
            fs_model = BSSWSS(k=self.num_features-i)  # remain 2 features.
            fs_model.fit(data, labels)
            prob = svm_problem(labels.tolist(),
                               fs_model.transform(data).tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            tmp_tv_inst = fs_model.transform(tv_inst)
            p1, p2, p3 = svm_predict(
                tv_labels, tmp_tv_inst.tolist(), model)

            accuracy = p2[0]
            precision = metrics.precision_score(tv_labels, p1)
            recall = metrics.recall_score(tv_labels, p1)
            f1_score = metrics.f1_score(tv_labels, p1)

            acc_list[i][0] = accuracy
            acc_list[i][1] = precision
            acc_list[i][2] = recall
            acc_list[i][3] = f1_score
            # if(tmp_f1_score-f1_score > tmp_f1_score*self.decline_rate):
            #     break
            # else:
            #     tmp_f1_score = f1_score
            tmp_f1_score = f1_score
        print(str(np.argmax(acc_list[:, 0]))+"："+str(acc_list[:, 0].max()))
        print(str(np.argmax(acc_list[:, 1]))+"："+str(acc_list[:, 1].max()))
        print(str(np.argmax(acc_list[:, 2]))+"："+str(acc_list[:, 2].max()))
        print(str(np.argmax(acc_list[:, 3]))+"："+str(acc_list[:, 3].max()))
        # f1_list=acc_list[:, 3].T.tolist()
        # draw_hist(f1_list," ","k","f1-score",0,len(f1_list),0,1)
        fs_model = BSSWSS(k=self.num_features-np.argmax(acc_list[:, 3]))
        fs_model.fit(data, labels)
        return fs_model
    
    def transform(self, data):
        return self.fs_model.transform(data)

def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    name_list = list(range(len(myList)))
    plt.figure()
    # name_list.reverse()
    rects = plt.plot(myList)
    # X轴标题
    index = list(range(len(myList)))
    # index = [float(c)+0.4 for c in range(len(myList))]
    plt.ylim(ymax=Ymax, ymin=Ymin)
    plt.xticks(index, name_list)
    plt.ylabel(Ylabel)  # X轴标签
    plt.xlabel(Xlabel)
    # for rect in rects:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height,
    #              str(height), ha='center', va='bottom')
    plt.title(Title)
    plt.show()
