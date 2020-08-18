clc
clear all
close all
IMA = im2double(imread('img2.jpeg'));
MOD = reshape(IMA,size(IMA,1)*size(IMA,2),3);                
clu     = 2;                                            
ce = MOD( ceil(rand(clu,1)*size(MOD,1)) ,:);             
d   = zeros(size(MOD,1),clu+2);                         
oo   = 15;
%%
% K-means
tic
for n = 1:oo
for i = 1:size(MOD,1)
for j = 1:clu  
d(i,j) = norm(MOD(i,:) - ce(j,:));      
end
[di,new] = min(d(i,1:clu));               
d(i,clu+1) = new;                                
d(i,clu+2) = di;                          
end
for i = 1:clu
A = (d(:,clu+1) == i);                          
ce(i,:) = mean(MOD(A,:));                           
end
end
X = zeros(size(MOD));
for i = 1:clu
idx = find(d(:,clu+1) == i);
X(idx,:) = repmat(ce(i,:),size(idx,1),1); 
end
T = reshape(X,size(IMA,1),size(IMA,2),3);
toc
%%
figure()
 imshow(IMA)
 title('image')
figure
 imshow(T); 
 title('decomposed')

