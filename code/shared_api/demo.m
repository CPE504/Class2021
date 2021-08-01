% x = [-8 -7 -6 -5 -4 -3, -2, 0, 2, 3, 4, 5, 6, 7, 8];
x = 1:10;
clc;
figure(1);clf(1)
y = softmax(x);
disp(sum(y))
% stem(x,y,'.');hold on;plot(x,y,'r');
y = softmin(x);
disp(sum(y))
% stem(x,y,'.');hold on;plot(x,y,'g');
[y, dy] = lsig_std(x);
stem(x,y,'-.','filled');hold on;plot(x,y,'-',x,dy,'-.');
% % stem(x,y);hold on;plot(x,dy,'-.');
y = softmin(x) - softmax(x); 
% y = -y;
disp(y)
stem(x,y,'-.','filled');hold on;plot(x,y,'k');
% f = cumsum(y);
g = gradient(gradient(y));
disp(g)
stem(x',g','-.','filled');hold on;plot(x',g');
axis padded; grid