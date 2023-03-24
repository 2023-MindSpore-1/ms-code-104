import numpy as np
from ecoc.BasePLECOC import BasePLECOC
# from sklearn.svm import SVC as libsvm
# from svmutil import *
from libsvm import svmutil
import torch.nn.functional as F
from libsvm.svmutil import *
from ecoc.GetComplexity import *
from sklearn import preprocessing
from ecoc.PreKNN import PreKNN, PLFeatureSelection
from ecoc.CodeMatrix.Matrix_tool import _exist_same_col, _exist_same_row, _exist_two_class
import torch
import torch.nn.functional as F


class RandPLECOC(BasePLECOC):

    def __init__(self, estimator=libsvm, max_iter=2000, **params):
        BasePLECOC.__init__(self, estimator, **params)
        self.max_iter = max_iter
        self.num_class = None
        self.codingLength = None
        self.min_num_tr = None
        self.coding_matrix = None
        self.models = None
        self.performance_matrix = None
        self.params = params
        self.fs_models = []
        self.plfs=None

    def create_integrity_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        # self.codingLength = int(np.ceil(10 * np.log2(self.num_class))) #38的长度啊，怪不得 编码长度长一点
        self.codingLength = 128#128 0.441 77 0.42 30->0.3989/0.4227
        self.min_num_tr = int(np.ceil(0.1 * num_tr))#self.min_num_tr 450 4599*0.1 设置了一个正负样本相加的阈值一定要大于这个

        coding_matrix = None
        final_coding_matrix=None
        counter = 0
        tmp_counter=0
        tmp_counter_iter=int(np.ceil(self.codingLength/4))#31
        tr_pos_idx = []
        tr_neg_idx = []
        final_tr_pos_idx=[]
        final_tr_neg_idx=[]
        tr_data_flag=np.zeros(num_tr) #(4498,)

        # test code start
        # csv_path = 'csv/matrix_dump.csv'
        # if os.path.exists(csv_path):
        #     os.remove(csv_path)
        # test_code_matrix = pd.read_csv(csv_path, header=-1).values
        # for i in range(test_code_matrix.shape[1]):
        # test code end
        for i in range(self.max_iter):#0->2000初始化的2000次
            tmpcode = np.int8(np.random.rand(self.num_class) > 0.5) #对于样本生成随机的13个的0/1串 
            if final_coding_matrix is not None:
                tmp_code_matrix = np.vstack((final_coding_matrix, tmpcode))#tmp直接放入final之后了
                while _exist_same_row(tmp_code_matrix):#如果有相同的就重新生成
                    tmpcode = np.int8(np.random.rand(self.num_class) > 0.5)
                    tmp_code_matrix = np.vstack((final_coding_matrix, tmpcode))
            # tmpcode = test_code_matrix[:, i]
            tmp_pos_idx = []
            tmp_neg_idx = []
            tmp_tr_data_flag=tr_data_flag.copy()#(4498,)
            for j in range(num_tr):
                # tempidx = np.argwhere(tr_labels[:, j]==1)
                # print('tempidx',tempidx, 'aaaaa')
                if np.all((tr_labels[:, j] & tmpcode) == tr_labels[:, j]):#看起来应该是partial label
                    tmp_pos_idx.append(j)#符合的就是正样本
                    tmp_tr_data_flag[j]+=1#第j个样本部分的flage再加1
                else:
                    if np.all((tr_labels[:, j] & np.int8(np.logical_not(tmpcode))) == tr_labels[:, j]):
                        tmp_neg_idx.append(j) #完全相反的就是负样本，负样本标注一下
                        tmp_tr_data_flag[j]+=1
            num_pos = len(tmp_pos_idx)
            num_neg = len(tmp_neg_idx)

            if (num_pos+num_neg >= self.min_num_tr) and (num_pos >= 5) and (num_neg >= 5) and (len(np.where(tmp_tr_data_flag==0)[0])==0 or len(np.where(tmp_tr_data_flag==0)[0])<len(np.where(tr_data_flag==0)[0])):
                tmp_counter = tmp_counter + 1
                tr_pos_idx.append(tmp_pos_idx)
                tr_neg_idx.append(tmp_neg_idx)
                coding_matrix = tmpcode if coding_matrix is None else np.vstack(
                    (coding_matrix, tmpcode))
                # tr_data_flag=tmp_tr_data_flag
                #符合阈值和条件 coding_matrix上加上 pos和neg上也添加好索引

            if tmp_counter == tmp_counter_iter: #迭代完了
                tmp_counter=0
                high_score_list=self.plfs.matrix_test(coding_matrix,tr_pos_idx,tr_neg_idx)
                if self.codingLength-counter<len(high_score_list):
                    high_score_list=high_score_list[:self.codingLength-counter]
                counter+=len(high_score_list)
                final_coding_matrix= coding_matrix[high_score_list,:] if final_coding_matrix is None else np.vstack(
                    (final_coding_matrix,coding_matrix[high_score_list,:]))
                coding_matrix=None
                for item in high_score_list:
                    final_tr_pos_idx.append(tr_pos_idx[item])
                    final_tr_neg_idx.append(tr_neg_idx[item])
                    for index in tr_pos_idx[item]:
                        tr_data_flag[index]+=1
                    for index in tr_neg_idx[item]:
                        tr_data_flag[index]+=1
                tr_pos_idx=[]
                tr_neg_idx=[]
        
            if counter >= self.codingLength:
                self.codingLength=counter
                break
        if counter != self.codingLength:
            raise ValueError(
                'The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        # dump_matrix = pd.DataFrame(coding_matrix.T)
        # dump_matrix.to_csv(csv_path, index=False, header=False)
        final_coding_matrix = (final_coding_matrix * 2 - 1).T #2-1=1 0*2-1=-1这样就变成正负一了
        print(len(np.where(tr_data_flag==0)[0]))
        return final_coding_matrix, final_tr_pos_idx, final_tr_neg_idx
        # return coding_matrix, tr_pos_idx, tr_neg_idx

    def create_coding_matrix(self, tr_data, tr_labels):
        num_tr = tr_data.shape[0]
        self.num_class = tr_labels.shape[0]
        self.codingLength = int(np.ceil(10 * np.log2(self.num_class)))
        self.min_num_tr = int(np.ceil(0.1 * num_tr))

        coding_matrix = None
        counter = 0
        tr_pos_idx = []
        tr_neg_idx = []
        tr_data_flag=np.zeros(num_tr)
        # complexityList=[]

        # test code start
        # csv_path = 'csv/matrix_dump.csv'
        # if os.path.exists(csv_path):
        #     os.remove(csv_path)
        # test_code_matrix = pd.read_csv(csv_path, header=-1).values
        # for i in range(test_code_matrix.shape[1]):
        # test code end
        for i in range(self.max_iter):
            tmpcode = np.int8(np.random.rand(self.num_class) > 0.5)
            # tmpcode = test_code_matrix[:, i]
            tmp_pos_idx = []
            tmp_neg_idx = []
            for j in range(num_tr):
                if np.all((tr_labels[:, j] & tmpcode) == tr_labels[:, j]):
                    tmp_pos_idx.append(j)
                else:
                    if np.all((tr_labels[:, j] & np.int8(np.logical_not(tmpcode))) == tr_labels[:, j]):
                        tmp_neg_idx.append(j)
            num_pos = len(tmp_pos_idx)
            num_neg = len(tmp_neg_idx)

            if (num_pos+num_neg >= self.min_num_tr) and (num_pos >= 5) and (num_neg >= 5):
                print(num_pos+num_neg)
                counter = counter + 1
                tr_pos_idx.append(tmp_pos_idx)
                tr_neg_idx.append(tmp_neg_idx)
                for item in tmp_pos_idx:
                    tr_data_flag[item]=1
                for item in tmp_neg_idx:
                    tr_data_flag[item]=1
                coding_matrix = tmpcode if coding_matrix is None else np.vstack(
                    (coding_matrix, tmpcode))
                
            if counter == self.codingLength:
                break

        if counter != self.codingLength:
            raise ValueError(
                'The required codeword length %s not satisfied', str(self.codingLength))
            self.codingLength = counter
            if counter == 0:
                raise ValueError('Empty coding matrix')
        # dump_matrix = pd.DataFrame(coding_matrix.T)
        # dump_matrix.to_csv(csv_path, index=False, header=False)
        coding_matrix = (coding_matrix * 2 - 1).T
        
        
        return coding_matrix, tr_pos_idx, tr_neg_idx

    def create_fs_base_models(self, tr_data, tr_pos_idx, tr_neg_idx, num_feature,tv_data,tv_labels):
        models = []
        # self.complexity=[]
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))
            tr_labels = np.hstack(
                (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
            # temp=getDataComplexitybyCol(tr_inst,tr_labels)
            # self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)

            # 使用PLFS
            # plfs = PLFeatureSelection(tr_data,tv_data,tv_labels)
            fs_model=self.plfs.fit(tr_inst, tr_labels,self.coding_matrix[:,i])
            self.fs_models.append(fs_model)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(),
                               fs_model.transform(tr_inst).tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            models.append(model)
        return models

    def create_base_models(self, tr_data, tr_pos_idx, tr_neg_idx):
        models = []
        # self.complexity=[]
        for i in range(self.codingLength):
            pos_inst = tr_data[tr_pos_idx[i]]
            neg_inst = tr_data[tr_neg_idx[i]]
            tr_inst = np.vstack((pos_inst, neg_inst))#样本特征
            tr_labels = np.hstack(
                (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))#正负1的label
            # temp=getDataComplexitybyCol(tr_inst,tr_labels)
            # self.complexity.append(temp)
            # model = self.estimator().fit(tr_inst, tr_labels)

            # # 使用PLFS
            # plfs = PLFeatureSelection(num_feature)
            # plfs.fit(tr_inst, tr_labels,tv_data,tv_labels)
            # self.fs_models.append(plfs)
            # libsvm 使用的训练方式
            prob = svm_problem(tr_labels.tolist(),
                               tr_inst.tolist())
            param = svm_parameter(self.params.get('svm_param'))
            model = svm_train(prob, param)
            models.append(model)
        return models

    def create_performance_matrix(self, tr_data, tr_labels):
        performance_matrix = np.zeros((self.num_class, self.codingLength))
        for i in range(self.codingLength):
            model = self.models[i]
            # p_labels = model.predict(tr_data)
            test_label_vector = np.ones(tr_data.shape[0])
            p_labels, _, _ = svm_predict(
                test_label_vector, self.fs_models[i].transform(tr_data).tolist(), model)
            p_labels = [int(i) for i in p_labels]
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                performance_matrix[j, i] = np.abs(sum(label_class_j[label_class_j ==
                                                                    self.coding_matrix[j, i]])/label_class_j.shape[0])
        return performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

    def create_base_performance_matrix(self, tr_data, tr_labels):
        performance_matrix = np.zeros((self.num_class, self.codingLength)) #13*矩阵长度
        for i in range(self.codingLength):
            model = self.base_models[i]#每一个svm判断是不是
            # p_labels = model.predict(tr_data)
            test_label_vector = np.ones(tr_data.shape[0]) #多少个样本
            p_labels, _, _ = svm_predict(
                test_label_vector, tr_data.tolist(), model) #根据data判断出是正样本还是负样本
            p_labels = [int(i) for i in p_labels]
            # print('p_labels shape', len(p_labels))
            # print(p_labels)
            # print('==================================================================================================')
            for j in range(self.num_class):
                label_class_j = np.array(p_labels)[tr_labels[j, :] == 1]
                # print('label_class_j', label_class_j)
                performance_matrix[j, i] = np.abs(sum(label_class_j[label_class_j ==
                                                                    self.coding_matrix[j, i]])/label_class_j.shape[0])
        return performance_matrix / np.transpose(np.tile(performance_matrix.sum(axis=1), (performance_matrix.shape[1], 1)))

    def fit(self, tr_data, tr_labels):
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
            tr_data, tr_labels)
        self.tr_pos_idx = tr_pos_idx
        self.tr_neg_idx = tr_neg_idx
        self.models = self.create_base_models(
            tr_data, tr_pos_idx, tr_neg_idx, tr_data.shape[1])
        self.performance_matrix = self.create_performance_matrix(
            tr_data, tr_labels)
        print(self.performance_matrix.shape)

    def fit_predict(self, tr_data, tr_labels, ts_data, ts_labels,tv_data,tv_labels, pre_knn, cntwrite):
        self.plfs = PLFeatureSelection(tr_data,tr_labels,tv_data,tv_labels,self.params)
        self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_integrity_coding_matrix(tr_data, tr_labels)#训练集的数据和partial label
        #产生了最终的编码矩阵和正负样本索引
        # print('self.coding_matrix',self.coding_matrix.shape)#13,38???
        # print('tr_pos_idx', len(tr_pos_idx))#38
        # print('hhhhhhh')
        # # self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
        # #     tr_data, tr_labels)
        self.tr_pos_idx = tr_pos_idx
        self.tr_neg_idx = tr_neg_idx
        # # repeat=int(tr_data.shape[1]/3)
        # # if(repeat>15):
        # #     repeat=15
        temp = []
        self.base_models = self.create_base_models(
            tr_data, tr_pos_idx, tr_neg_idx)
        print('self.base_models', self.base_models)
        # # self.models = self.create_fs_base_models(
        # #     tr_data, tr_pos_idx, tr_neg_idx, tr_data.shape[1],tv_data,tv_labels)
        self.base_performance_matrix = self.create_base_performance_matrix(
            tr_data, tr_labels)#13*38
        # self.performance_matrix = self.create_performance_matrix(
        #     tr_data, tr_labels)
        # print(self.performance_matrix.shape)
        matrix, base_accuracy,base_com_1_accuracy = self.base_predict(
            ts_data, ts_labels,pre_knn, cntwrite) #测试集数据和测试集0/1标签
        # print('matrix', matrix.shape)#matrix (13, 500)
        # print('base_accuracy',base_accuracy)#base_accuracy 0.654
        # print('base_com_1_accuracy',base_com_1_accuracy)#base_com_1_accuracy 0.654
        # # matrix, base_fs_accuracy, knn_accuracy, com_1_accuracy, com_2_accuracy = self.predict(
        # #     ts_data, ts_labels, pre_knn)
        # print('--------------------------------')
        temp.append(base_accuracy),
        temp.append(base_com_1_accuracy)
        # temp.append(base_fs_accuracy) 
        # temp.append(knn_accuracy)
        # temp.append(com_1_accuracy)
        # temp.append(com_2_accuracy)
        return temp
    def predict(self, ts_data, ts_labels, pre_knn):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            fs_model = self.fs_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, fs_model.transform(ts_data).tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        # count_common=0
        # for i in range(len(common_list)):
        #     if(np.array_equal(common_list[i],temp_common)):
        #         count_common+=1
        # print(count_common)
        # for i in range(ts_data.shape[0]):
        #     bin_pre_tmp = bin_pre[:, i]
        #     decision_pre_tmp = decision_pre[:, i]
        #     for j in range(self.num_class):
        #         code = self.coding_matrix[j, :]
        #         common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
        #         if(j==pre_labels[i]):
        #             self.data_decode.set_cols_list(np.where(common==0),np.where(common!=0))
        #         error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
        #         output_value[j, i] = -sum(common)-sum(error)

        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        base_accuracy = count / ts_data.shape[0]

        print(base_accuracy)

        _,knn_accuracy,pre_knn_matrix=pre_knn.predict(ts_data,ts_labels)
        print(knn_accuracy)

        tv_data,tv_labels=pre_knn.getValidationData()
        _,tv_knn_accuracy,knn_matrix=pre_knn.predict(tv_data,tv_labels)
        ecoc_matrix=self.fs_base_predict(tv_data,tv_labels)
        weight=pre_knn.getWeight(knn_matrix,ecoc_matrix)
        output_1_value = output_value*weight+pre_knn_matrix*(1-weight)
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_1_value[:, i] == max(output_1_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_1_accuracy = count / ts_data.shape[0]
        print(com_1_accuracy)

        output_2_value = pre_knn_matrix*output_value
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_2_value[:, i] == max(output_2_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_2_accuracy = count / ts_data.shape[0]
        print(com_2_accuracy)

        # pre_knn_matrix=pre_knn.getPreKnnMatrix()
        # output_value=output_value+pre_knn_matrix*0.5
        # pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        # for i in range(ts_data.shape[0]):
        #     idx = output_value[:, i] == max(output_value[:, i])
        #     pre_label_matrix[idx, i] = 1

        # count = 0
        # for i in range(ts_data.shape[0]):
        #     max_idx1 = np.argmax(pre_label_matrix[:, i])
        #     max_idx2 = np.argmax(ts_labels[:, i])
        #     if max_idx1 == max_idx2:
        #         count = count+1
        # com_accuracy = count / ts_data.shape[0]
        # print(com_accuracy)

        return pre_label_matrix, round(base_accuracy, 4), round(knn_accuracy, 4), round(com_1_accuracy, 4), round(com_2_accuracy, 4)

    def base_validation_predict(self, ts_data,ts_labels):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.base_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))

        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.base_performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.base_performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        base_accuracy = count / ts_data.shape[0]

        return pre_label_matrix

    def repredict(self, ts_data):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1
        return pre_label_matrix

    def refit_predict(self, tr_data, tr_labels, ts_data, ts_labels, pre_accuracy):
        coding_matrix = self.coding_matrix
        codingLength = self.codingLength
        self.accuracyList = []
        # 测试前10列
        for i in range(codingLength):
            tr_pos_idx = self.tr_pos_idx.copy()
            tr_neg_idx = self.tr_neg_idx.copy()
            self.coding_matrix = coding_matrix
            # 移除列
            tr_pos_idx.remove(tr_pos_idx[i])
            tr_neg_idx.remove(tr_neg_idx[i])
            temp = coding_matrix.transpose().tolist()
            temp.remove(temp[i])
            self.coding_matrix = numpy.array(temp).transpose()
            self.codingLength = codingLength-1
            self.models = self.create_base_models_no_complexity(
                tr_data, tr_pos_idx, tr_neg_idx)
            self.performance_matrix = self.create_performance_matrix(
                tr_data, tr_labels)
            pre_label_matrix, accuracy = self.predict(ts_data, ts_labels)
            self.accuracyList.append(accuracy)

        # 比较前10列
        posCol = []
        negCol = []
        for i in range(codingLength):
            if self.accuracyList[i]-pre_accuracy <= 0:
                posCol.append(i)
            else:
                negCol.append(i)
        print("积极列：")
        for item in posCol:
            print(
                str(self.accuracyList[item]-pre_accuracy)+" "+str(self.complexity[item]))
        print("消极列：")
        for item in negCol:
            print(
                str(self.accuracyList[item]-pre_accuracy)+" "+str(self.complexity[item]))

    def reshape(self, times, tr_data, tr_labels):
        coding_matrix = []
        for i in range(times):
            print(tr_labels.shape[0])
            temp_coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
                tr_data.copy(), tr_labels.copy())
            temp_complexity_list = []
            for i in range(self.codingLength):
                pos_inst = tr_data[tr_pos_idx[i]]
                neg_inst = tr_data[tr_neg_idx[i]]
                tr_inst = np.vstack((pos_inst, neg_inst))
                tr_labels = np.hstack(
                    (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))
                temp_complexity = getDataComplexitybyCol(tr_inst, tr_labels)
                temp_complexity_list.append(temp_complexity)
            print(temp_complexity_list)
            # f1_mean=np.array(temp_complexity_list).mean(axis=1)
            # print(f1_mean)
    def fs_base_predict(self,ts_data,ts_labels):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.models[i]
            fs_model = self.fs_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, fs_model.transform(ts_data).tolist(), model)
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))

        output_value = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i]
            for j in range(self.num_class):
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        # count_common=0
        # for i in range(len(common_list)):
        #     if(np.array_equal(common_list[i],temp_common)):
        #         count_common+=1
        # print(count_common)
        # for i in range(ts_data.shape[0]):
        #     bin_pre_tmp = bin_pre[:, i]
        #     decision_pre_tmp = decision_pre[:, i]
        #     for j in range(self.num_class):
        #         code = self.coding_matrix[j, :]
        #         common = np.int8(bin_pre_tmp == code) * self.performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
        #         if(j==pre_labels[i]):
        #             self.data_decode.set_cols_list(np.where(common==0),np.where(common!=0))
        #         error = np.int8(bin_pre_tmp != code) * self.performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
        #         output_value[j, i] = -sum(common)-sum(error)

        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_value[:, i] == max(output_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        base_accuracy = count / ts_data.shape[0]

        return pre_label_matrix

    def base_predict(self, ts_data,ts_labels,pre_knn,cntwrite):
        bin_pre = None
        decision_pre = None
        for i in range(self.codingLength):
            model = self.base_models[i]
            test_label_vector = np.ones(ts_data.shape[0])
            p_labels, _, p_vals = svm_predict(
                test_label_vector, ts_data.tolist(), model)#这边应该是正负1或者01的
            # p_labels = model.predict(ts_data)
            # p_vals = model.decision_function(ts_data)
            # p_vals = model.score(ts_data)

            bin_pre = p_labels if bin_pre is None else np.vstack(
                (bin_pre, p_labels))#判定样本的label
            decision_pre = np.array(p_vals).T if decision_pre is None else np.vstack(
                (decision_pre, np.array(p_vals).T))#判断为这个的置信度

        output_value = np.zeros((self.num_class, ts_data.shape[0]))#13*500的一个全0矩阵
        # print(bin_pre)
        # print(bin_pre.shape)#(38, 500)全是正负1的矩阵 500是有500个数据
        # print('====================================================================================================================')
        # tempp = None
        labelset = []
        aa = np.load('testpartialsomany.npy')#labelset200*3144
        bb = np.load('simmalirtweight.npy')
        size_aa = aa.shape[1]
        for i in range(0,size_aa):
            labelset.append(np.where(aa[:,i] == 1)[0]) 

        for i in range(ts_data.shape[0]):
            bin_pre_tmp = bin_pre[:, i]
            decision_pre_tmp = decision_pre[:, i] #第i个数据的标签预测和置信度
            for j in range(self.num_class):#13*128后面的数字可以任意选择，这里循环取每一类的编码matrix
                code = self.coding_matrix[j, :]
                common = np.int8(
                    bin_pre_tmp == code) * self.base_performance_matrix[j, :] / np.exp(np.abs(decision_pre_tmp))
                error = np.int8(
                    bin_pre_tmp != code) * self.base_performance_matrix[j, :] * np.exp(np.abs(decision_pre_tmp))
                output_value[j, i] = -sum(common)-sum(error)

        output_value = preprocessing.MinMaxScaler().fit_transform(output_value)
        # if cntwrite ==2:
        np.save('tempprobab', output_value)
        
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):#对于每一个样本
            # exp_x = np.exp(output_value[:, i])
            # softmax_x = exp_x / np.sum(exp_x)
            # print('output_value', softmax_x)
            idx = output_value[:, i] == max(output_value[:, i])#根据置信度找最大的一个类的索引确定类
            for ai in range(len(labelset[i])):
                labelidx = labelset[i][ai]
                output_value[labelidx,i] = output_value[labelidx,i] + bb[i, ai]
            pre_label_matrix[idx, i] = 1

        count = 0
        tensorsoft = torch.tensor(output_value)
        prosoft = F.softmax(tensorsoft, dim=0)
        simecocclass = torch.argmax(prosoft,dim=0).cpu().numpy()
        np.save('simecocclass', simecocclass)#ecoc+sim决定的类别
        classtemp = None
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])#这里就能获得类，就是分不清楚的类！！！0/1
            # print('max_indx1',pre_label_matrix[:, i] )
            if classtemp is None:
                classtemp = max_idx1
            else:
                classtemp = np.vstack((classtemp, max_idx1))
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        base_accuracy = count / ts_data.shape[0]
        # if cntwrite ==2:
        np.save('partialmedlabel1223', classtemp)#ecoc决定的类别

        _,_,pre_knn_matrix=pre_knn.predict(ts_data,ts_labels)

        tv_data,tv_labels=pre_knn.getValidationData()
        _,tv_knn_accuracy,knn_matrix=pre_knn.predict(tv_data,tv_labels)
        ecoc_matrix=self.base_validation_predict(tv_data,tv_labels)
        weight=pre_knn.getWeight(knn_matrix,ecoc_matrix)
        output_1_value = output_value*weight+pre_knn_matrix*(1-weight)
        pre_label_matrix = np.zeros((self.num_class, ts_data.shape[0]))
        for i in range(ts_data.shape[0]):
            idx = output_1_value[:, i] == max(output_1_value[:, i])
            pre_label_matrix[idx, i] = 1

        count = 0
        for i in range(ts_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(ts_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        com_1_accuracy = count / ts_data.shape[0]
        print('com_1_accuracy',com_1_accuracy)

        return pre_label_matrix,round(base_accuracy, 4),round(com_1_accuracy, 4)

    def pic_4_1(self, tr_data, tr_labels, ts_data, ts_labels,tv_data,tv_labels, pre_knn):
        self.plfs = PLFeatureSelection(tr_data,tr_labels,tv_data,tv_labels,self.params)
        coding_matrix, tr_pos_idx, tr_neg_idx = self.create_integrity_coding_matrix(
            tr_data, tr_labels)
        # self.coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
        #     tr_data, tr_labels)
        
        # repeat=int(tr_data.shape[1]/3)
        # if(repeat>15):
        #     repeat=15
        f1_list_1,acc_list_1=self.plfs.matrix_test_4_1(coding_matrix,tr_pos_idx,tr_neg_idx)

        coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
            tr_data, tr_labels)
        
        f1_list_2,acc_list_2=self.plfs.matrix_test_4_1(coding_matrix,tr_pos_idx,tr_neg_idx)
        myList=[]
        myList.append(f1_list_2)
        myList.append(f1_list_1)
        draw_hist_4_1(myList," ","Col","f1-score",0,len(f1_list_1),0,1)
        myList=[]
        myList.append(acc_list_2)
        myList.append(acc_list_1)
        draw_hist_4_1(myList," ","Col","accuracy",0,len(f1_list_1),0,1)
    def exl_4_1(self, tr_data, tr_labels, ts_data, ts_labels,tv_data,tv_labels, pre_knn):
        self.plfs = PLFeatureSelection(tr_data,tr_labels,tv_data,tv_labels,self.params)
       
        for i in range(10):
            coding_matrix, tr_pos_idx, tr_neg_idx = self.create_coding_matrix(
            tr_data, tr_labels)
        
 
def draw_hist_4_1(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    name_list = list(range(len(myList[0])))
    plt.figure()
    # name_list.reverse()
    total_width, n = 0.8, 2  
    width = total_width / n 
    x=list(range(len(myList[0])))
    rects1 = plt.bar(x, myList[0],label='PL-ECOC',width=width, fc = 'y')
    for i in range(len(x)):  
        x[i] = x[i] + width  
    rects2 = plt.bar(x, myList[1],label='PL-ECOC-1',width=width, fc = 'r')
    # X轴标题
    index = list(range(len(myList[0])))
    # index = [float(c)+0.4 for c in range(len(myList))]
    plt.ylim(ymax=Ymax, ymin=Ymin)
    plt.xticks(index, name_list)
    plt.ylabel(Ylabel)  # X轴标签
    plt.xlabel(Xlabel)
    # for rect in rects1:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height,
    #              str(height), ha='center', va='bottom')
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height,
    #              str(height), ha='center', va='bottom')             
    plt.title(Title)
    plt.legend()
    # plt.savefig("pictures/"+file_name+".png")
    plt.show()

    