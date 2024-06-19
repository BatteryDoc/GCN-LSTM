import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

# In[]
#数据提取
#root:数据的地址
#fl:数据组的名称
#number:电芯编号
#singal_size:节点长度
#Q_0:电流切换点对应的SOC    
        
root ='D:/data/data_new/matlab_data1_21_48.mat'
fl = loadmat(root)['batch4']
number = 7
singal_size = 10
Initial_capacity = 1.1
Q_0 = Initial_capacity*0.27

# 加载对应的电池数据，数据类型包括电池的电压，充电时间，充电量和充电容量
def data_load(fl,number):
    
    V,Q,C,T = [],[],[],[]
    
    for i in range(0, len(fl[number][2][0])):
    
         V_1 = []
         t_1 = []
         C_1 = np.array(fl[number][2][0][i][8])
         Q_1 = []
         
         for j in range(0,len(fl[number][2][0][i][3])):
             if fl[number][2][0][i][3][j] >= 0:
                 V_2 = fl[number][2][0][i][4][j]
                 t_2 = fl[number][2][0][i][1][j]
                 Q_2 = fl[number][2][0][i][2][j]
                 
                 V_1.append(V_2)
                 t_1.append(t_2)
                 Q_1.append(Q_2)
                     
         V_1 = np.array(V_1)
         Q_1 = np.array(Q_1)
         V.append(V_1)
         T.append(t_1)
         C.append(C_1)
         Q.append(Q_1)
         
    return V,Q,C,T

V,Q,C,T = data_load(fl, number)
# In[]
#数据清洗
def clean_and_sort_data(T, V, Q):
    
    t_cleaned, V_cleaned, Q_cleaned = [],[],[]
    
    for m in range(len(V)):
        
        data = np.array([T[m], V[m], Q[m]]).T      
        data = data.squeeze()
        data = data[data[:, 0].argsort()]
        
        _, unique_indices = np.unique(data[:, 0], return_index=True)
        data = data[unique_indices]
        
        t_cleaned.append(data[:, 0])
        V_cleaned.append(data[:, 1])
        Q_cleaned.append(data[:, 2])
        
       
    return t_cleaned, V_cleaned, Q_cleaned

t_cleaned,V_cleaned,Q_cleaned = clean_and_sort_data(T, V, Q)

#提取电压片段
#寻找电流切换点   
for o in range(len(Q_cleaned)):
    
    closet_Q_0 = min(Q_cleaned[o], key=lambda x: abs(x - Q_0))
    index_1 = np.where(Q_cleaned[o] == closet_Q_0)[0][0] 
    t_cleaned[o] = t_cleaned[o] -t_cleaned[o][index_1]
    
in_t = np.linspace(-2,2,50)

interpolated_V = []
interpolated_Q = []

from scipy.interpolate import interp1d
for m in range(len(V_cleaned)):
    # Create linear interpolators
    if len(t_cleaned[m]) != len(V_cleaned[m]) or len(t_cleaned[m]) != len(Q[m]):
        print(f"在周期 {m} 中，数组长度不匹配: t = {len(t_cleaned[m])}, V = {len(V_cleaned[m])}, Q = {len(Q_cleaned[m])}")
        continue
    
    if len(np.unique(t_cleaned[m])) != len(t_cleaned[m]):
        print(f"在周期 {m} 中，时间数组t含有重复值。")
        continue

    try:
        # 创建线性插值函数
        interp_V = interp1d(t_cleaned[m], V_cleaned[m], kind='linear', bounds_error=False, fill_value='extrapolate')
        # interp_Q = interp1d(t_clean[m], Q_clean[m], kind='linear', bounds_error=False, fill_value='extrapolate')

        # 在定义的时间网格上进行插值
        interpolated_V.append(interp_V(in_t))
        # interpolated_Q.append(interp_Q(in_t))
    except Exception as e:
        print(f"在周期 {m} 中发生错误：{str(e)}")
# In[]
#划分图的节点数据和节点标签与全局标签  
def data_load(interpolated_V,C,singal_size):
    
    data = []#节点数据
    lab = []#全局标签
    node_lab = []#节点标签
    start, end = 0,  singal_size
    
    for i in range (0,len(interpolated_V)):
        
        data_row = []
        lab_row = C[i]
        node_lab_row = []
        
        while end <= 50 :
            
            data_row.append(interpolated_V[i][start:end])
            node_lab_row.append(C[i])
            start += singal_size
            end += singal_size
            
        data.append(data_row)
        node_lab.append(node_lab_row)
        lab.append(lab_row)
        start, end = 0, singal_size
        
    return data,lab,node_lab

data,lab,node_lab = data_load(interpolated_V,C,singal_size)
# In[]
#将图数据划分为训练集和测试集

def get_train_data(data,node_lab,lab):
    
    data_array = np.array(data)
    node_lab_array = np.array(node_lab)
    lab_array = np.array(lab)
    mean = np.mean(data_array)
    std = np.std(data_array)
    normalized_data = (data_array - mean) / std

    X_train, X_test, y_train, y_test ,lab_train, lab_test= train_test_split(normalized_data, node_lab_array, lab_array, test_size=0.40, random_state=9)#42#15#9
    
    #对齐循环的次数
    def get_original_indices(initial_data, extracted_data):
        
        original_indices = []
        
        for item in extracted_data:
            index = np.where(initial_data == item)[0][0]
            original_indices.append(index)
            
        original_indices = np.array(original_indices)
        
        return original_indices

    original_indices_y_test = get_original_indices(node_lab_array, y_test)
    original_indices_y_train = get_original_indices(node_lab_array, y_train)
    
    return X_train,X_test,y_train,y_test,lab_train, lab_test,original_indices_y_test,original_indices_y_train

X_train,X_test,y_train,y_test,lab_train,lab_test,original_indices_y_test,original_indices_y_train = get_train_data(data,node_lab,lab)
# In[]
sort_indices_2 = np.argsort(original_indices_y_test.flatten())
sorted_indices_y_test = original_indices_y_test[sort_indices_2]
lab_test = lab_test[sort_indices_2]

# In[]
#生成图数据
def Gen_edge(data_array):#生成边和边的权重
    data_array = np.array(data_array)
    A_data = []
    edge_indices_data = []
    values_list_data = []
    for i in range(data_array.shape[0]):
        attr = torch.Tensor(data_array[i])
        attr = attr.view(attr.size(0), -1)
        A1 = torch.mm(attr, attr.T)

        maxval, maxind = A1.max(axis=1)  
        A1_norm = A1 / maxval  
        k = A1.shape[0]  
        values, indices = A1_norm.topk(k, dim=1, largest=True,sorted=False) 
        edge_index_i = torch.tensor([[], []], dtype=torch.long)
        edge_values_i = torch.tensor([])
        
        for j in range(indices.shape[0]):
            index_1 = torch.zeros(indices.shape[1], dtype=torch.long) + j
            index_2 = indices[j]

            # 创建自身连接的掩码
            self_loop_mask = (index_1 != index_2)

            # 使用掩码排除自身连接
            index_1 = index_1[self_loop_mask]
            index_2 = index_2[self_loop_mask]
            values_j = values[j][self_loop_mask]

            sub_index = torch.stack([index_1, index_2])
            edge_index_i = torch.cat([edge_index_i, sub_index], axis=1)

            # 只保留前 k 个值，以匹配相应的边
            values_list_i = values_j[:k]
            edge_values_i = torch.cat([edge_values_i, values_list_i])

        A_data.append(A1_norm)
        edge_indices_data.append(edge_index_i)
        values_list_data.append(edge_values_i)
    return A_data,edge_indices_data,values_list_data
    
A_data,edge_indices_data,values_list_data = Gen_edge(data)
# In[]

#生成全部图

data_graph_data = []
for i in range(len(data)):
    graph_idx = i
    x = torch.Tensor(data[graph_idx])
    edge_index = edge_indices_data[graph_idx]
    edge_attr = values_list_data[graph_idx]
    node_labels = torch.Tensor(node_lab[graph_idx])
    label = torch.Tensor(lab[graph_idx])
    data_1_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_1_data.data = node_labels
    data_1_data.y = label
    data_graph_data.append(data_1_data)

#生成训练图的边索引和权重
#------------------------------------------------------------------------
A,edge_indices,values_list = Gen_edge(X_train)
#------------------------------------------------------------------------
# In[]
#生成训练图
#------------------------------------------------------------------------------
data_graph_train = []
for i in range(X_train.shape[0]):
    graph_idx = i
    x = torch.Tensor(X_train[graph_idx])
    edge_index = edge_indices[graph_idx]
    edge_attr = values_list[graph_idx]
    node_labels = torch.Tensor(y_train[graph_idx])
    label = torch.Tensor(lab_train[graph_idx])
    data_1_train = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_1_train.data = node_labels
    data_1_train.y = label
    data_graph_train.append(data_1_train)
#生成测试图图的边索引和权重
A_test,edge_indices_test,values_list_test = Gen_edge(X_test)

#生成测试图
data_graph_test = []
for j in range(X_test.shape[0]):
    graph_idx = j
    x = torch.Tensor(X_test[graph_idx])
    edge_index = edge_indices_test[graph_idx]
    edge_attr = values_list_test[graph_idx]
    node_labels = torch.Tensor(y_test[graph_idx])
    y = torch.Tensor(lab_test[graph_idx])
    data_1_test = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data_1_test.data = node_labels
    data_1_test.y = y
    data_graph_test.append(data_1_test)

#构建GCN+LSTM模型

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCNModel, self).__init__()

        # 图卷积层
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim,40)

        # LSTM层
        self.lstm = nn.LSTM(input_size=40, hidden_size=10, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(10, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # 第二层图卷积
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # LSTM层
        x, _ = self.lstm(x.unsqueeze(0))  

        # 连接层，将节点信息汇总成一个节点
        x = x.mean(dim=0, keepdim=True)

        # 全连接层
        x = self.fc1(x)

        return x

num_features = singal_size 
hidden_dim = 80 

model = GCNModel(num_features, hidden_dim)

# 创建 DataLoader
train_loader = DataLoader(data_graph_train, batch_size=1, shuffle=True)

# 定义损失函数为均方误差 (MAE) 损失
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        # 计算绝对误差
        absolute_errors = torch.abs(predicted - target)

        # 计算平均绝对误差
        mae = torch.mean(absolute_errors)

        return mae
        
# 创建 MAE 损失函数
criterion = MAELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0025)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    for data in train_loader:
        data.x = data.x.squeeze(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(train_loader)}')

import os
save_path = 'models/model.pth'
torch.save(model.state_dict(), save_path)
save_dir = os.path.dirname(save_path)

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 保存模型参数
torch.save(model.state_dict(), 'model.pth')

# 创建测试 DataLoader
test_loader = DataLoader(data_graph_test, batch_size=1, shuffle=False)
model.load_state_dict(torch.load('model.pth'))

# 列表用于存储模型的预测值和实际值
predictions_test = []
actual_values_test = []

# 测试模型
model.eval() 
with torch.no_grad():
    for data in test_loader:
        data.x = data.x.squeeze(-1)
        output = model(data)
        predictions_test.append(output.tolist())
        actual_values_test.append(data.y.tolist())

# 展开实际值列表，因为 data.y 是列表的列表
actual_values_test = [y for sublist in actual_values_test for y in sublist]
actual_values_test = np.array(actual_values_test)/ Initial_capacity
actual_values_test_SOH = actual_values_test * 100

predictions_test = np.array(predictions_test)
predictions_test = predictions_test.squeeze() 
predictions_test = predictions_test.mean(axis=1) / Initial_capacity
predictions_test = predictions_test.reshape(-1, 1)
predictions_test_SOH = predictions_test * 100
predictions_test = predictions_test[sort_indices_2]
predictions_test_SOH = predictions_test_SOH[sort_indices_2]

error_test = actual_values_test - predictions_test
abs_error_test = np.abs(error_test)
MAE = float(np.mean(abs_error_test))
RMSE = float(np.sqrt(mean_squared_error(actual_values_test,predictions_test)))

# # 计算均方根误差 (RMSE)
MAE = "{:.6f}".format(MAE)
RMSE = "{:.6f}".format(RMSE)
print(f'MAE: {MAE}')
print(f'RMSE: {RMSE}')

# In[]
# 可视化
from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':20,
        }
rcParams.update(params)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=0.45)

plt.plot(sorted_indices_y_test, actual_values_test_SOH ,label='Real' ,color = '#D7191C',linewidth=2,marker='*',alpha=0.6,markersize=8)
plt.plot(sorted_indices_y_test, predictions_test_SOH ,label='GCN+LSTM' ,color = '#2C7BB6',linestyle='--',linewidth=2,marker='o',alpha=0.4,markersize=8)
plt.xlabel('Cycle')
plt.ylabel('SOH (%)')
plt.ylim(75, 100)  
plt.legend(fontsize=18)
plt.show()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes,Bbox
from scipy.stats import norm


fig, ax = plt.subplots()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=0.45)
color_values = predictions_test_SOH - actual_values_test_SOH  
scatter = plt.scatter( actual_values_test_SOH,predictions_test_SOH, c=color_values, cmap='coolwarm', s=55, label='GCN+LSTM')

cbar = plt.colorbar(scatter)
cbar.set_label('Error (%)', rotation=270, labelpad=15)
plt.legend(fontsize=18)
plt.plot([75, 100], [75, 100], color='black', linewidth=2)
plt.xlabel('Observed SOH (%)')
plt.ylabel('Estimated SOH (%)')
plt.ylim(75, 100)
plt.xlim(75, 100)

bbox = Bbox.from_bounds(-0.4, -0.85, 1.4, 1.4)
axins = inset_axes(ax, width="25%", height="25%",bbox_to_anchor=bbox,bbox_transform=ax.transAxes)
axins.hist(color_values, bins=20, edgecolor='black', color='gray', density=True)
mu, std = norm.fit(color_values)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
axins.plot(x, p, 'k', linewidth=3)
axins.set_xlabel('Error (%)', fontsize=14)  
axins.set_ylabel('Density', fontsize=14) 
axins.tick_params(axis='both', labelsize=10) 
plt.show()
