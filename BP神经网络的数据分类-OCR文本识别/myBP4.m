clc
clear all
close all
tic
%% 训练数据预测数据提取及归一化

% 下载四类语音信号
load HW1503data X y
[row, col] = size(X);
classNum = 10;
trainNum = 4000;
testNum = 1000;

% 输入输出数据,第1维为类别标识，后24维为语音特征信号
input = ones(row, col+1);
input(:,2:end) = X;

outputClass = y;
output = zeros(1, classNum * row);
output(classNum .* (0:row-1)' + outputClass) = 1;
output = (reshape(output, [classNum, row]))';

% 随机提取4000个样本为训练样本，1000个样本为预测样本
nPerm = randperm(row); % 从1到5000间随机排序
input_train = input(nPerm(1 : trainNum), :)';
output_train = output(nPerm(1 : trainNum), :)';
input_test = input(nPerm(trainNum+1 : row), :)';
output_test = output(nPerm(trainNum+1 : row), :)';

% 输入数据归一化
[inputn,inputps] = mapminmax(input_train);

%% 网络结构初始化
inNum = col+1;
midNum = 25;
outNum = classNum;
 
% 权值初始化
epsilonInit = sqrt(6) / sqrt(inNum + outNum);
W10 = rands(midNum, inNum) * epsilonInit;
W21 = rands(outNum, midNum) * epsilonInit;

% 学习率
eta = 0.01;

%% 网络训练
iterMax = 200;
eIter = zeros(iterMax, 1);
for iter = 1:iterMax
    for n = 1:trainNum
        % 取一个样本
        oneIn = inputn(:, n);
        oneOut = output_train(:, n);
        
        % 隐藏层输出             
        hOut = 1 ./ (1 + exp(- W10 * oneIn));

        % 输出层输出
        yOut = W21 * hOut;
        
        % 计算误差
        eOut = oneOut - yOut;     
        eIter(iter) = eIter(iter) + sum(abs(eOut));
        
        % 计算输出层误差项 delta2
        delta2 = eOut;
        
        % 计算隐藏层误差项 delta1
        FI = hOut .* (1 - hOut);
        delta1 = (FI .* (W21' * eOut));
        
        % 更新权重
        W21 = W21 + eta * delta2 * hOut';
        W10 = W10 + eta * delta1 * oneIn';
    end
end
 

%% 测试
inputn_test = mapminmax('apply', input_test, inputps);

% 隐藏层输出 
hOut = 1 ./ (1 + exp(- W10 * inputn_test));

% 输出层输出
fore = W21 * hOut;

%% 结果分析
% 根据网络输出找出数据属于哪类
[output_fore, ~] = find(bsxfun(@eq, fore, max(fore)) ~= 0);

%BP网络预测误差
error = output_fore' - outputClass(nPerm(trainNum+1 : row))';

%% 计算正确率
% 找出每类判断错误的测试样本数
kError = zeros(1, classNum);  
outPutError = bsxfun(@and, output_test, error);
[indexError, ~] = find(outPutError ~= 0);

for class = 1:classNum
    kError(class) = sum(indexError == class);
end

% 找出每类的总测试样本数
kReal = zeros(1, classNum);
[indexRight, ~] = find(output_test ~= 0);
for class = 1:classNum
    kReal(class) = sum(indexRight == class);
end

% 正确率
rightridio = (kReal-kError) ./ kReal
meanRightRidio = mean(rightridio)
%}

%% 画图

% 画出误差图
figure
stem(error, '.')
title('BP网络分类误差', 'fontsize',12)
xlabel('信号', 'fontsize',12)
ylabel('分类误差', 'fontsize',12)

% 画目标函数
figure
plot(eIter)
title('每次迭代总的误差', 'fontsize', 12)
xlabel('迭代次数', 'fontsize', 12)
ylabel('所有样本误差和', 'fontsize', 12)

