clc
x=subject.Train.train1;
T=x(1,2)-x(1,1);
fs=1/T;
L=length(x(1,:));
a = filter(d,x(3,:));
Y = fft(a);
Y(1)=0;
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*((0:(L/2))/L);
plot(f,P1) ;
title('Single-Sided Amplitude Spectrum of X(t)');
xlabel('f (Hz)');
ylabel('|P1(f)|');

%%
load lp_filter
load SubjectData1.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train1=train
subject.Test.test1=test


load SubjectData2.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train2=train
subject.Test.test2=test

load SubjectData3.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train3=train
subject.Test.test3=test

load SubjectData4.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train4=train
subject.Test.test4=test

load SubjectData5.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train5=train
subject.Test.test5=test

load SubjectData6.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train6=train
subject.Test.test6=test

load SubjectData7.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train7=train
subject.Test.test7=test

load SubjectData8.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train8=train
subject.Test.test8=test

load SubjectData9.mat;
for i = 1 : 9 
   train(i,:) = filtfilt(d,train(i,:));
   test(i,:) = filtfilt(d,test(i,:));
end
subject.Train.train9=train
subject.Test.test9=test
%%

[f1,f2]=IndexExtraction(subject);
subject.Time=f1;

%%
function [yy,ep]=IndexExtraction(x)

A=x.Test;
B=x.Train

te=fieldnames(A);
tr=fieldnames(B);
o1=0;
o2=0;
o3=0;
for i=1:9
    o1=0;
    o2=0;
    o3=0;
    
    q=B.(tr{i});
    r= size(q,2);
   
    w=A.(te{i});
    t=size(w,2);
    
    for k=1:t
        
       if w(10,k)~=0 && w(10,k-1)==0
           o1=o1+1; 
          u1(1,o1)=k;
          m1(:,:,o1)=w(:,k-50:k+203);
          
          
       end
        
    end
    
y1(i).Time.Test=u1;
          
epo(i).Test=m1;

   for j=1:r
       
    if q(10,j)~=0 && q(10,j-1)==0 && q(11,j)==1
         
          o2=o2+1;
          u2(1,o2)=j;
          m2(:,:,o2)=q(:,j-50:j+203);
          
   
    
    elseif q(10,j)~=0 && q(10,j-1)==0 && q(11,j)==0
            
        
          o3=o3+1;
          u3(1,o3)=j;
          m3(:,:,o3)=q(:,j-50:j+203);
      
     
        end
   end

    y1(i).Time.Target=u2;    
     
    epo(i).Taeget=m2; 
   
     
     y1(i).Time.Non_Target=u3;
      
     epo(i).Non_Taeget=m3;
   
end
   
   

 ep=epo;  
   
 yy=y1;  
   
   
end

function gd = groupdelay(h,N)
n = [0:length(h)-1]
jh_prime = n .* h;
gd = real(fft(jh_prime,N) ./ fft(h,N)); 
end


function y = zphasefilter(h,x)
N = 8000;
yy = filter(h,1,x);
gd = round(groupdelay(h,N));
gd = gd(abs(gd) < 10000);
gd = gd(~isnan(gd));
gd = gd(~isinf(gd));
gd_valid = round(mean(gd));
y = yy(gd_valid + 1 : end);
end
