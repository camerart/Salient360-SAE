mu=0;                      %均值

sigma=1;                  %标准差，方差的开平方

b=sigma/sqrt(2);      %根据标准差求相应的b

x=rand(1,2000)-0.5;
% x=linspace(-1.25, 1.25, 100);    %生成(-0.5,0.5)区间内均匀分布的随机数列 (一万个数的行向量);

y=exp(-sign(x-mu).*(x-mu)/b)/(2*b) + 0.01*(rand(1,2000)-0.5); %生成符合拉普拉斯分布的随机数列
std(x)
mean(x)
figure;
scatter(x,y, 15,'filled')
ylabel('logh(\theta)');
xlabel('\theta');
grid;